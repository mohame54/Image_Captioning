
import collections
import configparser as Config
import json
from preprocess_inputs import *
import time
from Image_caption import *


def train(Epochs, ds, model, b_loss, path_to_save):
    losses, accs = [], []
    for epoch in range(1, Epochs + 1):
        st = time.time()
        total_loss, total_acc = 0, 0
        print(f'Epoch: {epoch}/{Epochs}')
        for i, batch in enumerate(ds):
            loss, acc = model.train_step(batch)
            total_loss += loss
            total_acc += acc
            if loss <= b_loss:
                b_loss = loss
                print(f"Saved the model with loss:{b_loss:.3f}.")
                model.save_weights(path_to_save)
            if i % 100 == 0:
                print(f'Epoch:{epoch},batch:{i},loss: {loss:.3f} acc: {acc:.3f}.')
        total_loss /= len(ds)
        total_acc /= len(ds)
        print(f'Finished epoch:{epoch} in {(time.time() - st):.3f} sec,loss: {total_loss:.3f},acc: {total_acc:.3f}.\n ')
        losses.append(total_loss)
        accs.append(total_acc)
    return {'epochs': [i for i in range(Epochs)], 'losses': losses, 'accs': accs}


def get_params(config):
    c = Config.ConfigParser()
    c.read(config)
    default = c['Default']
    hyper = c['HYPERs']
    data_pars = c['DATASET_PARAMS']
    return default, hyper, data_pars


def load_annotations(path_dict):
    with open(path_dict['annotation_file']) as f:
        anns = json.load(f)
    path = path_dict['image_folder']
    image_path_to_caption = collections.defaultdict(list)
    for val in anns['annotations']:
        caption = f"<start> {val['caption']} <end>"
        image_path = path + '%012d.jpg' % (val['image_id'])
        image_path_to_caption[image_path].append(caption)
    return image_path_to_caption


def main():
    config_file = 'config.ini'
    default, hypers, dataset_params = get_params(config_file)
    mapping_dict = load_annotations(default)
    caption_processor = CaptionProcessor(mapping_dict, dataset_params['num_samples'])
    tokenizer = caption_processor.get_tokenizer()
    feats, labels, _ = caption_processor.get_features_labels()
    train_ds = make_ds(feats, labels, dataset_params['batch_size'], tokenizer)
    embed_dim, ffd, num_heads, ratio = hypers
    cnn = get_cnn_model()
    aug = get_aug_model()
    vocab = tokenizer.get_vocabulary()
    img_cap = ImageCaption(cnn, num_heads, embed_dim,
                           ffd, ratio, aug, vocab)
    loss_obj = tf.losses.SparseCategoricalCrossentropy(reduction='none')
    num_train_steps = len(train_ds) * int(dataset_params['epochs'])
    num_warmup_steps = num_train_steps // 15
    lr = LRSchedule(1e-3, num_warmup_steps)
    Adam = tf.optimizers.Adam(lr)
    img_cap.compile(Adam, loss=loss_obj)
    hist = train(dataset_params['epochs'], train_ds, img_cap, 1.2, default['path_to_save'])


if __name__ == '__main__':
    main()
