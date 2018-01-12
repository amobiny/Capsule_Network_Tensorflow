import argparse

parser = argparse.ArgumentParser()

# For margin loss
parser.add_argument('--m_plus', default=0.9, help='m+ parameter')
parser.add_argument('--m_minus', default=0.1, help='m- parameter')
parser.add_argument('--lambda_val', default=0.5, help='Down-weighting parameter for the absent class')

# For reconstruction loss
parser.add_argument('--alpha', default=.392, help='Regularization coefficient to scale down the reconstruction loss')
parser.add_argument('--mode', default='train', help='train or test')
parser.add_argument('--batch_size', default=128, help='Batch size')
parser.add_argument('--epoch', default=50, help='Total number of training epochs')
parser.add_argument('--iter_routing', default=3, help='Number of routing iterations')
parser.add_argument('--stddev', default=0.01, help='dataset name, mnist or fashion-mnist')

parser.add_argument('--dataset', default='mnist', help='std for W initializer')
parser.add_argument('--n_cls', default=10, help='Total number of classes')
parser.add_argument('--img_w', default=28)
parser.add_argument('--img_h', default=28)
parser.add_argument('--n_ch', default=1, help='Number of input image channels')

parser.add_argument('--restore_training', default=True, help='Restores the last trained model to resume training')
parser.add_argument('--checkpoint_path', default='./saved_model/', help='path for saving the model checkpoints')
parser.add_argument('--log_dir', default='./log_dir/', help='logs directory')
parser.add_argument('--results', default='./results/', help='path for saving the results')
parser.add_argument('--save_freq', default=1, help='The frequency of saving the model (epoch)')
parser.add_argument('--save_hdf5', default=True, help='To save the train/test results in separate HDF5 files')
parser.add_argument('--tr_disp_sum', default=100, help='The frequency of saving train results (step)')
args = parser.parse_args()
