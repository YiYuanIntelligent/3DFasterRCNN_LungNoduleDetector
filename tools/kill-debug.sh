kill -9 `ps aux|grep train_net.py | awk '{print $2}'`
