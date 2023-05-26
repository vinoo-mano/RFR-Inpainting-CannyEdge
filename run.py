import argparse
import os
from model import RFRNetModel
from dataset import Dataset
from torch.utils.data import DataLoader

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--mask_root', type=str)
    parser.add_argument('--model_save_path', type=str, default='checkpoint')
    parser.add_argument('--result_save_path', type=str, default='results')
    parser.add_argument('--target_size', type=int, default=256)
    parser.add_argument('--mask_mode', type=int, default=1)
    parser.add_argument('--num_iters', type=int, default=2500)
    parser.add_argument('--model_path', type=str, default="/home/jupyter/RFR-Inpainting/checkpoint/checkpoint_paris.pth")
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--n_threads', type=int, default=6)
    parser.add_argument('--resume', type=str)  # Add resume argument
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--test', action='store_true')
    
    parser.add_argument('--gpu_id', type=str, default="0")
    args = parser.parse_args()
        
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    model = RFRNetModel()
    if args.test:
        model.initialize_model(args.model_path, False)
        model.cuda()
        dataloader = DataLoader(Dataset(args.data_root, args.mask_root, args.mask_mode, args.target_size, mask_reverse=True, training=False))
        print(f"Loaded {len(dataloader)} images and masks")
        model.test(dataloader, args.result_save_path)
    else:
        if args.resume:
            model.initialize_model(args.resume, True)
            model.cuda()
            print("In Run.py file data root" + (args.data_root))
            #print("In Run.py file mask root" + (args.mask_root))
            dataloader = DataLoader(Dataset(args.data_root, args.mask_root, args.mask_mode, args.target_size, mask_reverse=True), batch_size=args.batch_size, shuffle=True, num_workers=args.n_threads)
            print(f"Loaded {len(dataloader)} images and masks")
            model.train(dataloader, args.model_save_path, args.finetune, args.num_iters)
        else:
            print("No resume path provided. Exiting.")
            exit()

if __name__ == '__main__':
    run()

    
    #python run.py --test --model_path checkpoint/checkpoint.pth --result_save_path results/test_results
    #python run.py --data_root RFR-Inpainting/places2/train_256_places365standard/data_256 --model_save_path RFR-Inpainting/checkpoint --model_path RFR-Inpainting/checkpoint/checkpoint_paris.pth --num_iters 2500 --batch_size 6 --gpu_id -1
    
    #python run.py --data_root /home/jupyter/RFR-Inpainting/Places2/Places2/train_256_places365standard/data_256 --mask_root RFR-Inpainting/ --model_save_path RFR-Inpainting/ --result_save_path RFR-Inpainting/ --batch_size 6 --n_threads 4 --num_iters 2500 --gpu_id -1
    
    #python run.py --data_root /home/jupyter/RFR-Inpainting/places2/train_256_places365standard/data_256 --model_save_path RFR-Inpainting/ --result_save_path RFR-Inpainting/ --batch_size 6 --n_threads 4 --num_iters 2500 --gpu_id -1
    
    #python run.py --data_root /home/jupyter/RFR-Inpainting/places2/train_256_places365standard/data_256 --model_save_path RFR-Inpainting/ --result_save_path RFR-Inpainting/ --batch_size 6 --n_threads 4 --num_iters 2500 --gpu_id -1 --resume RFR-Inpainting/checkpoint/checkpoint_paris.pth
    
    #python run.py --data_root /home/jupyter/RFR-Inpainting/places2/train_256_places365standard/data_256 --model_save_path /home/jupyter/RFR-Inpainting/checkpoint --result_save_path /home/jupyter/RFR-Inpainting/checkpoint  --batch_size 6 --n_threads 4 --num_iters 2500 --gpu_id -1 --resume /home/jupyter/RFR-Inpainting/checkpoint/checkpoint_paris.pth
    
    

    
'''
    else:
        if args.resume:
            model.initialize_model(args.resume, True)
        else:
            model.initialize_model(args.model_path, True)
        #model.initialize_model(args.model_path, True)
        #model.initialize_model(args.resume or args.model_path, True)  # Pass the resume flag
        model.cuda()
        dataloader = DataLoader(Dataset(args.data_root, args.mask_root, args.mask_mode, args.target_size, mask_reverse = True), batch_size = args.batch_size, shuffle = True, num_workers = args.n_threads)
        print(f"Loaded {len(dataloader)} images and masks")
        model.train(dataloader, args.model_save_path, args.finetune, args.num_iters)
'''

