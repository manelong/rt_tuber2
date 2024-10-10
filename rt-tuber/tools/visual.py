'''使用onnx对模型进行推理，对一个文件夹内的视频进行可视化，绘制结果并保存下来'''
import torch

import numpy as np 
import onnxruntime as ort 

import cv2
from collections import deque

import os
import time

def main(args, ):
    """main
    """
    video_files = os.listdir(args.video_file)
    output_files = args.output_file
    if not os.path.exists(output_files):
        os.makedirs(output_files)

    # 加载模型
    sess = ort.InferenceSession(args.onnx_file, providers=['CUDAExecutionProvider'])
    print(ort.get_device())

    # im_data = torch.randn(1,6 , 3, 288, 512)
    # orig_size = torch.tensor([[1080, 1920]])
    for k, video_file in enumerate(video_files):

        video_path = os.path.join(args.video_file, video_file)  
        # 读取视频,定义一个deque(长度为K)，用来存储视频的帧
        cap = cv2.VideoCapture(video_path)
        frames_deque = deque(maxlen=args.K)
        # 获得视频的帧率和分辨率
        fps = cap.get(cv2.CAP_PROP_FPS)

        # 定义一个保存视频
        output_path = os.path.join(output_files, video_file)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (1920, 1080))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = frame
            orig_size = torch.tensor([1920, 1080]).unsqueeze(0)
            # print(orig_size)
            frame_resize = cv2.resize(frame_rgb, (512, 288))
            frames_deque.append(frame_resize)
            f = np.array(frames_deque)
            frame_ = torch.tensor(f).permute(0, 3, 1, 2).unsqueeze(0).float()
            if len(frames_deque) == args.K:
                start_time = time.time()
                output = sess.run(
                    # output_names=['labels', 'boxes', 'scores'],
                    output_names=None,
                    input_feed={'images': frame_.data.numpy(), "orig_target_sizes": orig_size.data.numpy()}
                )
                print('Inference time: {}'.format(time.time() - start_time))
                labels, boxes, scores = output

                # 绘制结果
                out_frame = cv2.resize(frame, (1920, 1080))
                for i in range(args.top_K):
                    bbox = boxes[0][i][-4:]
                    score = scores[0][i]

                    # 绘制视频
                    if score > args.thr:
                        out_frame = cv2.rectangle(out_frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                        out_frame = cv2.putText(out_frame, str(score), (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                out.write(out_frame)

        cap.release()
        out.release()
        print(f'video {k+1}/{len(video_files)} done!')
        
    # draw([im_pil], labels, boxes, scores)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx-file', type=str, default='/home/sports/data/code/code_hub/basketball_moc/rt_tuber_code/model.onnx')
    parser.add_argument('--video_path', type=str, default='/home/sports/data/code/code_hub/basketball_moc/rt_tuber_code/data/newbasketball/test_video/huilongguan_400w_20231221_2100-2130_0001_101.1_102.4.mp4')
    parser.add_argument('-video_file', type=str, default='/home/sports/data/code/code_hub/basketball_moc/rt_tuber_code/test_data/test_1')
    parser.add_argument('-output_file', type=str, default='/home/sports/data/code/code_hub/basketball_moc/rt_tuber_code/test_data/reselt_test_1')
    parser.add_argument('-d', '--device', type=str, default='cuda')
    parser.add_argument('-K', '--K', type=int, default=6)
    parser.add_argument('-top_K', type=int, default=1, help='top K results to show')
    parser.add_argument('-thr', type=float, default=0.5, help='threshold to show')

    args = parser.parse_args()
    main(args)
