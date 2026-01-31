#cd /home/user/Documents/tensorrt_yolo26/build/ultralytics-8.4.0

#wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt -O yolo26n.pt
#wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt -O yolo26s.pt
#wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt -O yolo26m.pt
#wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l.pt -O yolo26l.pt
#wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x.pt -O yolo26x.pt

#python gen_wts.py -w yolo26n.pt -o yolo26n.wts -t detect
#python gen_wts.py -w yolo26s.pt -o yolo26s.wts -t detect
#python gen_wts.py -w yolo26m.pt -o yolo26m.wts -t detect
#python gen_wts.py -w yolo26l.pt -o yolo26l.wts -t detect
#python gen_wts.py -w yolo26x.pt -o yolo26x.wts -t detect

#rm *.pt
#mv *.wts /home/user/Documents/tensorrt_yolo26/build

cd /home/user/Documents/tensorrt_yolo26/build

./yolo26_det -s yolo26n.wts yolo26n.engine n
./yolo26_det -s yolo26s.wts yolo26s.engine s
./yolo26_det -s yolo26m.wts yolo26m.engine m
./yolo26_det -s yolo26l.wts yolo26l.engine l
./yolo26_det -s yolo26x.wts yolo26x.engine x