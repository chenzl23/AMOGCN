import argparse

def parameter_parser():
    parser = argparse.ArgumentParser(description = "Run .")

    parser.add_argument("--dataset-name", nargs = "?", default = "ACM3025.mat")
    parser.add_argument("--epoch-num", type = int, default = 500, help = "Number of training epochs. Default is 500.")
    parser.add_argument("--seed", type = int, default = 42, help = "Random seed for train-test split. Default is 42.")
    parser.add_argument("--dropout", type = float, default = 0.5, help = "Dropout parameter. Default is 0.5.")
    parser.add_argument("--learning-rate", type = float, default = 0.01, help = "Learning rate. Default is 0.01.")

    parser.add_argument('--isSemi', type=bool, default=True, help='(0/1) Include supervision component?')

    parser.add_argument("--verbose", type = int, default = 1, help = "Show training details.")

    # Dataspilt
    parser.add_argument("--train-ratio", type = float, default = 0.6, help = "Train data ratio. Default is 0.6.")
    parser.add_argument("--valid-ratio", type = float, default = 0.1, help = "Valid data ratio. Default is 0.2.")
    parser.add_argument("--test-ratio", type = float, default = 0.1, help = "Test data ratio. Default is 0.3.")

    parser.add_argument("--feature-normalize", type = int, default = 0, help = "If feature normalization")
    parser.add_argument("--k", type = int, default = 100, help = "k of KNN graph.")
    parser.add_argument("--gamma", type = float, default = 0.1, help = "parameter for semantic information.")


    # Parameters
    parser.add_argument("--layer-num", type = int, default = 1, help = "Layer number.")
    parser.add_argument("--hidden-dim", type = int, default = 16, help = "Layer number.")

    # for early stop
    parser.add_argument("--early-stop", type = bool, default = False, help = "If early stop")
    parser.add_argument("--patience", type = int, default = 20, help = "Patience for early stop")
    parser.add_argument("--lr-patience", type = int, default = 40, help = "Patience for learning rate adapt")

    
    return parser.parse_args()
