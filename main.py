from harmonize import *

def get_args():
    parser = argparse.ArgumentParser()
    # path relative to the directory ./
    parser.add_argument("--src_img_path", default="Con_Sin_GAN/Images/Harmonization/scream.jpg")
    parser.add_argument("--naive_img_path", default="Con_Sin_GAN/Images/Harmonization/scream_naive.jpg")
    parser.add_argument("--output_dir", default="Images/output")
    parser.add_argument("--min_size", default=120, type=int)
    parser.add_argument("--gpu", default=1, type=int)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    args.src_img_path = os.path.join("..", args.src_img_path)
    args.naive_img_path = os.path.join("..", args.naive_img_path)
    args.output_dir = os.path.join("..", args.output_dir)
    write_html(args)
    train_model(args)
    model_dir, src_img_name = harmoize(args)
    get_img(args, model_dir, src_img_name)