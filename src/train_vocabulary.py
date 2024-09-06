import argparse
import time
import torch
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from dataset import make_dataset, make_data_loader, process_dataset
from module import resume, to_device
from dataset.ibpe import BasicTokenizer
from module import check, resume, to_device, process_control


cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)


def main():
    process_control()
    run_experiment()
    return


def run_experiment():
    dataset = make_dataset(cfg['data_name'])
    dataset = process_dataset(dataset)
    model = BasicTokenizer()
    data_loader = make_data_loader(dataset, cfg[cfg['tag']]['optimizer']['batch_size'], cfg['num_steps'],
                                   cfg['step'], cfg['step_period'], cfg['pin_memory'], cfg['num_workers'],
                                   cfg['collate_mode'], cfg['seed'])
    data_iterator = enumerate(data_loader['train'])
    return


def train(data_loader, model):
    start_time = time.time()

    for i, input_data in data_loader:
        input_size = input_data['data'].size(0)
        input_data = to_device(input_data, cfg['device'])
        output = model(input_data)

    return


if __name__ == "__main__":
    main()



in_dir = 'data/MNIST/raw/train'  # change this according to your dataset directory
img_list = []
img_list.extend(glob.glob(os.path.join(in_dir, "*")))
print("training dataset size: ", len(img_list))

# create a directory for models, so we don't pollute the current directory
os.makedirs("models", exist_ok=True)

t0 = time.time()
for TokenizerClass, name in zip([BasicTokenizer], ["basic"]):
    # construct the Tokenizer object and kick off verbose training
    tokenizer = TokenizerClass()
    for in_img in img_list:
        image = Image.open(in_img)
        img_array = asarray(image)
        img_array_flat = img_array.flatten()
        tokenizer.train(img_array_flat, 256 + 100, 2, resume=True,
                        verbose=False)  # 256 are the byte tokens, then do ? merges
    prefix = os.path.join("models", name)
    tokenizer.save(prefix)  # writes two files in the models directory: name.model, and name.vocab
t1 = time.time()

print(f"Training took {t1 - t0:.2f} seconds")