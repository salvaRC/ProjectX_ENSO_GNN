from GNN2.layer import *


class gtnet(nn.Module):
    def __init__(self, gcn_true, buildA_true, gcn_depth, num_nodes, device, args, predefined_A=None, static_feat=None,
                 dropout=0.3, subgraph_size=20, node_dim=40, dilation_exponential=1, conv_channels=32,
                 residual_channels=32, skip_channels=64, end_channels=128, seq_length=12, in_dim=2, out_dim=12,
                 layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True):
        super(gtnet, self).__init__()
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.device = device
        self.args = args
        self.adj_matrix = None if predefined_A is None else torch.from_numpy(predefined_A).to(device).float()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        if args.prelu:
            self.final_act = [nn.PReLU().to(device), nn.PReLU().to(device)]
        else:
            self.final_act = [nn.ReLU().to(device), nn.ReLU().to(device)]
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha,
                                    static_feat=static_feat, use_prelu=args.graph_prelu)

        self.seq_length = seq_length
        kernel_size = 7
        if dilation_exponential > 1:
            self.receptive_field = int(
                1 + (kernel_size - 1) * (dilation_exponential ** layers - 1) / (dilation_exponential - 1))
        else:
            self.receptive_field = layers * (kernel_size - 1) + 1

        for i in range(1):
            if dilation_exponential > 1:
                rf_size_i = int(
                    1 + i * (kernel_size - 1) * (dilation_exponential ** layers - 1) / (dilation_exponential - 1))
            else:
                rf_size_i = i * layers * (kernel_size - 1) + 1
            new_dilation = 1
            for j in range(1, layers + 1):
                if dilation_exponential > 1:
                    rf_size_j = int(
                        rf_size_i + (kernel_size - 1) * (dilation_exponential ** j - 1) / (dilation_exponential - 1))
                else:
                    rf_size_j = rf_size_i + j * (kernel_size - 1)

                self.filter_convs.append(
                    dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(
                    dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))
                if self.seq_length > self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                     out_channels=skip_channels,
                                                     kernel_size=(1, self.seq_length - rf_size_j + 1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                     out_channels=skip_channels,
                                                     kernel_size=(1, self.receptive_field - rf_size_j + 1)))

                if self.gcn_true:
                    self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))

                if self.seq_length > self.receptive_field:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),
                                               elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),
                                               elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential

        self.layers = layers
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length),
                                   bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels,
                                   kernel_size=(1, self.seq_length - self.receptive_field + 1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels,
                                   kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1),
                                   bias=True)

        self.idx = torch.arange(self.num_nodes).to(device)
        self.mlp = ONI_MLP(input_dim=num_nodes, output_dim=1, device=device, use_prelu=args.prelu)

    def forward(self, input, idx=None):
        seq_len = input.size(3)
        assert seq_len == self.seq_length, 'input sequence length not equal to preset sequence length'
        if self.seq_length < self.receptive_field:
            input = nn.functional.pad(input, (self.receptive_field - self.seq_length, 0, 0, 0))
            # print("Input shape after padding:", input.shape)

        if self.gcn_true:
            if self.buildA_true:
                if idx is None:
                    self.adj_matrix = self.gc(self.idx)
                else:
                    self.adj_matrix = self.gc(idx)

        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        for i in range(self.layers):
            residual = x
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip
            if self.gcn_true:
                x = self.gconv1[i](x, self.adj_matrix) + self.gconv2[i](x, self.adj_matrix.transpose(1, 0))
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            if idx is None:
                x = self.norm[i](x, self.idx)
            else:
                x = self.norm[i](x, idx)

        skip = self.skipE(x) + skip
        x = self.final_act[0](skip)
        x = self.final_act[1](self.end_conv_1(x))
        x = self.end_conv_2(x)
        x = self.mlp(x.squeeze())
        return x

    def file_path(self):
        string = f"{self.args.horizon}lead_{self.args.index}_{self.args.window}w_" \
                 f"{self.args.normalize}normed_{self.args.resolution}res_{self.num_nodes}nodes_" \
                 f"{self.layers}layers_{self.gcn_depth}gcnDepth"
        return string

    def update_device(self, device):
        self.device = device
        self.gc.device = device


class ONI_MLP(nn.Module):
    """
    Fully connected MLP on top of node embeddings
    """

    def __init__(self, input_dim, output_dim, device, L=2, use_prelu=False):  # L=nb_hidden_layers
        super().__init__()
        FC_layers = [nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=True) for l in range(L)]
        FC_layers.append(nn.Linear(input_dim // 2 ** L, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(FC_layers)
        self.L = L
        self.activations = [nn.PReLU().to(device) for _ in range(L)] \
            if use_prelu else [nn.ReLU().to(device) for _ in range(L)]

    def forward(self, x):
        y = x
        for layer, act in zip(self.FC_layers, self.activations):
            y = layer(y)
            y = act(y)
        y = self.FC_layers[self.L](y)
        return y