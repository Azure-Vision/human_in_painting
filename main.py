from harmonize import *
from make_naive import *
def get_args():
    parser = argparse.ArgumentParser()
    # path relative to the directory ./
    parser.add_argument("--src_img_path", default="hwr_test/nightsky.jpeg")
    parser.add_argument("--human_img_path", default="hwr_test/tmp.jpg")
    parser.add_argument("--naive_img_path", default=None) # "Con_Sin_GAN/Images/Harmonization/scream_naive.jpg"
    parser.add_argument("--output_dir", default="Images/output")
    parser.add_argument("--min_size", default=120, type=int)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--no-finetune", default=False, action="store_true")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    args.input_name = args.src_img_path
    args.train_mode = "harmonization"
    if not args.naive_img_path:
        human_segmentation = segment_human(args.human_img_path)
        args.naive_img_path = get_naive_image(human_segmentation, args.src_img_path)
    args.naive_img_path = os.path.join("..", args.naive_img_path)
    args.src_img_path = os.path.join("..", args.src_img_path)
    args.output_dir = os.path.join("..", args.output_dir)
    write_html(args)
    train_model(args)
    model_dir, src_img_name = harmoize(args)
    get_img(args, model_dir, src_img_name)