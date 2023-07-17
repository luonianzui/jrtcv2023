# ----------------------------------------------
#
#           SCURM_Vision_Global_Config
#               Coding By Pikachu
#                  全局配置信息
#
#            LAST_UPDATE:NOV15/2020
#
# ----------------------------------------------
global_debug_flag = True   # 全局调试输出

camera_resol_high = 1024   # 设定图像高度
camera_resol_widt = 1280   # 设定图像宽度
camera_resol_sfps = 100    # 设定曝光帧率
# 若是Linux，BR输出需要反向，应设置为True
camera_resol_tbgr = False  # 反转B和R通道

serial_debug_flag = True   # 串口调试输出
serial_debug_data = False  # 输出串口内容
serial_verify_md5 = True   # 验证-MD5数据
serial_comst_head = [255, 255]   # 分隔符

shared_debug_flag = True   # 输出失败信息
shared_detail_inf = True   # 输出警告信息

debugs_write_flag = True   # 调试写入文件
# debugs_write_file = '/root/scurm/scurm.txt'
debugs_write_file = 'scurm.txt'
