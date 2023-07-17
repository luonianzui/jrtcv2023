# ----------------------------------------------
#
#           SCURM_Vision_Module_Serial
#               Coding By Pikachu
#                  串口传输模块
#
#            LAST_UPDATE: JAN17/2021
#
# ----------------------------------------------
import binascii
import hashlib
import struct
import base64
import time
import serial
import serial.tools.list_ports
import signal

from module import ModuleConfig as config
from module import ModuleDebugs as debugs

# 模块化配置---------------------------------------------
if config.serial_debug_flag and config.global_debug_flag:
    debug_flag = True
    debug_head = "Serial"
    debug_data = config.serial_debug_data
else:
    debug_flag = False
    debug_data = False


# 串口类对象------------------------------------------------------------------------------------------------------------
class Serial:
    # 列出全部串口 -----------------------------------------------------------------------------------------------------
    # 参数：无， 静态：是
    # 返回：<str-所有串口>
    @staticmethod
    def comList():
        pcgl_list = list(serial.tools.list_ports.comports())
        if debug_flag:
            if pcgl_list == 0:
                debugs.Debugs(debug_head, "NO SERIAL FIND!", 2)
            else:
                debugs.Debugs(debug_head, "SERIALS FOUND!!", 0)
                print("----------------------COM LISTS---------------------")
                for i in range(0, len(pcgl_list)):
                    print(pcgl_list[i])
                print("----------------------------------------------------")
        return pcgl_list

    # 列出数据内容 -----------------------------------------------------------------------------------------------------
    # 参数：<obj-数据来源>, <bool-显示全部>
    # 返回：<str-数据内容>，静态：静态方法
    @staticmethod
    def lstData(in_item, in_flag=False):
        lstitem_retu = []
        for lstitem_loop in in_item:
            if in_flag:
                lstitem_retu.append({
                    'name': lstitem_loop['name'],
                    'type': lstitem_loop['type'],
                    'flag': lstitem_loop['flag'],
                    'data': lstitem_loop['data']
                })
            else:
                lstitem_retu.append(lstitem_loop['name'])
        return lstitem_retu

    # 载入物理串口------------------------------------------------------------------------------------------------------
    # 参数：无，静态:否
    # 返回：<bool-结果>
    def comLoad(self):
        while self.lock:
            time.sleep(0.001)
        self.lock = True
        try:
            self.coms = serial.Serial(self.port, self.bbps, timeout=self.time)
            if debug_flag:
                debugs.Debugs(debug_head, "OPEN SERIAL OK!", 4)
                self.init = True
            self.lock = False
            return True
        except serial.serialutil.SerialException:
            if debug_flag:
                debugs.Debugs(debug_head, "NO SUCH SERIAL!", 2)
            self.lock = False
            return False

    # 设置串口信息------------------------------------------------------------------------------------------------------
    def comInfo(self, in_send=None, in_dest=None):
        while self.lock:
            time.sleep(0.001)
        self.lock = True
        if debug_flag:
            debugs.Debugs(debug_head, "SERIAL SETINFO!", 4)
        if in_send is None or in_dest is None:
            self.lock = False
            return self.send, self.dest
        else:
            self.send = in_send
            self.dest = in_dest
            self.comInit()
            self.lock = False
            return True

    # 初始化串口内容----------------------------------------------------------------------------------------------------
    def comInit(self):
        self.sedt['HeadID'] = bytes(config.serial_comst_head)
        self.sedt['SendID'] = self.send
        self.sedt['RecvID'] = self.dest
        if debug_flag:
            debugs.Debugs(debug_head, "SERIAL INITIAL!", 4)
        return True

    def comChek(self):
        if not self.init or not self.coms.isOpen():
            if debug_flag:
                debugs.Debugs(debug_head, "SERIAL NO INIT!", 2)
                debugs.Debugs(debug_head, "YOU MAY INIT IT", 1)
            return False
        return True

    def comSend(self):
        if not self.comChek():
            return False
        if len(self.seit) == 0:
            return False
        while self.lock:
            time.sleep(0.001)
        self.lock = True
        self.nums = self.nums + 1
        self.sedt['Packet'] = self.nums
        if debug_data:
            print(self.sedt)
        letsend_text = bytes()
        letsend_md5s = hashlib.md5()
        for llt_loop in self.seit:
            letsend_temp = None
            if llt_loop['type'] is None:
                if isinstance(llt_loop['data'], int):
                    letsend_temp = struct.pack('i', llt_loop['data'])
                    llt_loop['type'] = 'I'
                elif isinstance(llt_loop['data'], str):
                    letsend_temp = bytes(llt_loop['data'], encoding="ASCII")
                    llt_loop['type'] = 'S'
                elif llt_loop['flag'] == 'G':
                    letsend_temp = bytes("", encoding="ASCII")
                    llt_loop['type'] = 'G'
                elif llt_loop['data'] is None and llt_loop['flag'] == 'P':
                    if debug_flag:
                        debugs.Debugs(debug_head, "EMPTY DATA !!!!", 1)
                    continue
                else:
                    if debug_flag:
                        debugs.Debugs(debug_head, "UNKNOW TYPE:" + str(type(llt_loop['data'])), 1)
                    continue
            elif llt_loop['type'] in ['I', 'S', 'G']:
                try:
                    if llt_loop['flag'] == 'G':
                        letsend_temp = bytes('', encoding="ASCII")
                    elif llt_loop['type'] == 'I':
                        letsend_temp = bytes(llt_loop['data'].to_bytes(4, byteorder='little'))
                    elif llt_loop['type'] == 'S':
                        letsend_temp = bytes(llt_loop['data'], encoding="ASCII")
                    elif llt_loop['type'] == 'G':
                        letsend_temp = bytes('', encoding="ASCII")
                except AttributeError or BaseException:
                    debugs.Debugs(debug_head, "严重的数据类型错误！！", 4)
                    continue
            else:
                if debug_flag:
                    debugs.Debugs(debug_head, "UNKNOW TYPE:" + llt_loop['type'], 1)
                continue
            letsend_text = letsend_text + bytes(llt_loop['flag'] + llt_loop['type'], encoding="ASCII")
            letsend_text = letsend_text + struct.pack("i", len(letsend_temp))
            letsend_text = letsend_text + bytes(llt_loop['name'], encoding="ASCII") + letsend_temp
        letsend_md5s.update(letsend_text)
        if debug_data:
            print(letsend_md5s.hexdigest()[:20])
            print('包内内容：', '                                                 ' + str(letsend_text))
        letsend_send = self.sedt['HeadID'] + bytes([self.sedt['SendID'] % 255]) + bytes([self.sedt['RecvID'] % 255])
        letsend_base = struct.pack("i", self.sedt['Packet']) + struct.pack('i', len(letsend_text))
        letsend_base = letsend_base + bytes(letsend_md5s.hexdigest()[:20], encoding="ASCII") + letsend_text
        if debug_data:
            print('完整数据：', str(letsend_base))
        letsend_base = base64.b64encode(letsend_base)
        letsend_send = letsend_send + letsend_base
        if debug_data:
            print('BASE64后：', '                ' + str(letsend_base))
            print('全包输出：', str(letsend_send))
        # letsend_time = True
        # eventlet.monkey_patch()
        # with eventlet.Timeout(3, False):
        try:
            self.coms.write(letsend_send)
        except OSError or BaseException:
            debugs.Debugs("Serial", "串口物理错误！！", 3)
        # letsend_time = False
        # if letsend_time:
        #     debugs.Debugs(debug_head, "警告：系统报告串口发送超时", 1)
        #     debugs.Debugs(debug_head, "请检查下位机串口是否打开！", 1)
        self.lock = False
        return letsend_send

    # 串口接收数据------------------------------------------------------------------------------------------------------
    # 输入：<byte-外部传入数据(可选)>
    # 输出：<bool-是否成功获取到数据>
    def comRecv(self, in_byte=None):
        # 检查初始化和加锁状态---------------------------------------
        if not self.comChek():
            return False
        while self.lock:
            time.sleep(0.001)
        self.lock = True
        # 从外部还是从串口读取---------------------------------------
        if in_byte is None:
            comrecv_temp = self.coms.read_all()
        else:
            comrecv_temp = in_byte
        # 判断读到数据是否为空---------------------------------------
        if len(comrecv_temp) == 0:
            self.lock = False
            return False
        # 循环从字符串获取数据---------------------------------------
        comrecv_recv = None  # 记录数据的内容
        comrecv_retu = False  # 记录成功的状态
        for comrecv_byte in comrecv_temp:
            # --------------------------------------当头数据未被读入
            if self.flag < 2:
                # ----------------------------------跳过非0xff的数据
                if comrecv_byte != 0xff:
                    self.flag = 0
                    continue
                # ----------------------------------当识别了0xff数据
                else:
                    self.buff = self.buff + bytes([comrecv_byte])
                    self.flag = self.flag + 1
            # --------------------------------------已经读取头部数据
            else:
                if comrecv_byte != 0xff:
                    self.buff = self.buff + bytes([comrecv_byte])
                else:
                    self.flag = 1
                    comrecv_recv = self.buff
                    if self.readStr(comrecv_recv):
                        comrecv_retu = True
                    self.buff = bytes() + bytes([comrecv_byte])
        if self.flag == 2:
            comrecv_recv = self.buff
            if self.readStr(comrecv_recv):
                self.buff = bytes()
                self.flag = 0
                comrecv_retu = True
        self.lock = False
        return comrecv_retu

    def readStr(self, in_data):
        comrecv_recv = in_data
        comrecv_base = None
        if comrecv_recv is None or len(comrecv_recv) < 43:
            return False
        self.redt['HeadID'] = bytes(config.serial_comst_head)
        self.redt['SendID'] = comrecv_recv[2]
        self.redt['RecvID'] = comrecv_recv[3]
        if debug_data:
            print(comrecv_recv[4:])
        try:
            comrecv_base = bytes(base64.b64decode(comrecv_recv[4:]))
            comrecv_flag = True
        except binascii.Error:
            comrecv_flag = False
            if debug_data:
                debugs.Debugs(debug_head, "BASE64解码未能成功", 0)
        except BaseException or IOError:
            comrecv_flag = False
            if debug_data:
                debugs.Debugs(debug_head, "BASE64遇到严重错误", 1)
        if not comrecv_flag or len(comrecv_base) <= 28:
            return False
        if debug_data:
            print('完整数据：', str(comrecv_base))
        self.redt['Packet'] = struct.unpack('i', comrecv_base[:4])[0]
        if debug_data:
            print(self.redt)
        comrecv_leng = struct.unpack('i', comrecv_base[4:8])[0]
        if debug_data:
            print(comrecv_leng, len(comrecv_base[28:]))
            print('包内内容：', '                                                 ' + str(comrecv_base[28:]))
        if comrecv_leng == len(comrecv_base[28:]):
            comrecv_md5s = hashlib.md5()
            comrecv_md5s.update(comrecv_base[28:])
            # print(comrecv_md5s.hexdigest()[:20], comrecv_base[8:28])
            if not config.serial_verify_md5 or\
                    bytes(comrecv_md5s.hexdigest()[:20], encoding="ASCII") == comrecv_base[8:28]:
                comrecv_text = comrecv_base[28:]
                comrecv_iter = 0
                self.reit = []
                if debug_data:
                    print(type(comrecv_text))
                while comrecv_iter < len(comrecv_text):
                    # print(comrecv_iter)
                    iter_data = None
                    iter_flag = chr(comrecv_text[comrecv_iter])
                    iter_type = chr(comrecv_text[comrecv_iter + 1])
                    iter_lens = struct.unpack('i', comrecv_text[comrecv_iter + 2:comrecv_iter + 6])[0]
                    iter_name = str(comrecv_text[comrecv_iter + 6:comrecv_iter + 10])[2:-1]
                    if iter_flag == 'G':
                        iter_data = ''
                    elif iter_type == 'S':
                        iter_data = str(comrecv_text[comrecv_iter + 10:comrecv_iter + 10 + iter_lens])[2:-1]
                    elif iter_type == 'I':
                        iter_data = struct.unpack('i', comrecv_text[comrecv_iter + 10:comrecv_iter + 10 + iter_lens])
                    else:
                        if debug_flag:
                            debugs.Debugs(debug_head, "UNKNOW TYPE:" + iter_type, 1)
                        comrecv_iter = comrecv_iter + 10 + iter_lens
                        continue
                    if debug_data:
                        print(iter_flag, iter_type, iter_lens, iter_name, iter_data)
                    comrecv_iter = comrecv_iter + 10 + iter_lens
                    iter_temp = {
                        'name': iter_name,
                        'flag': iter_flag,
                        'data': iter_data,
                        'type': iter_type
                    }
                    self.reit.append(iter_temp)
                if debug_data:
                    print(self.reit)
            else:
                return False
        else:
            return False
        return True

    # --------------------------------------------------
    #
    #         添加发送的数据表项（发送变量和请求变量）
    #
    #       输入：变量名，传输类型，变量类型，数据内容
    # --------------------------------------------------
    def addSend(self,
                in_name,  # 串口变量名称 限制4字节
                in_flag="P",  # P是发出数据，G查询数据
                in_type=None,  # 初始化数据类型，可不填
                in_data=None  # 初始化数据内容，可不填
                ):
        if not self.comChek():
            return False
        while self.lock:
            time.sleep(0.001)
        self.lock = True
        additem_temp = {
            'name': in_name[:4],
            'flag': in_flag,
            'data': in_data,
            'type': in_type
        }
        self.seit.append(additem_temp)
        self.lock = False
        return True

    # ------------------------------------------------
    #
    #               删除某个待发送数据项
    #
    #                  输入：变量名
    # ------------------------------------------------
    def delSend(self, in_name):
        if not self.comChek():
            return False
        while self.lock:
            time.sleep(0.001)
        self.lock = True
        for delitem_loop in self.seit:
            if delitem_loop['name'] == in_name:
                self.seit.remove(delitem_loop)
                self.lock = False
                return True
        self.lock = False
        return False

    # ------------------------------------------------
    #
    #               返回所有待发送变量名
    #
    #        in_flag:True-输出类型，False-只输出名称
    # ------------------------------------------------
    def lstSend(self, in_flag=False):
        if not self.comChek():
            return False
        while self.lock:
            time.sleep(0.001)
        return self.lstData(in_item=self.seit,
                            in_flag=in_flag)

    # ------------------------------------------------
    #
    #              返回待发送某个变量的数据
    #
    #                   输入：变量名
    # ------------------------------------------------
    def getSend(self, in_name):
        return self.getData(self.seit, in_name)

    # ------------------------------------------------
    #
    #               修改某个待发送数据项
    #
    #      输入：变量名，数据内容，数据类型（可选）
    # ------------------------------------------------
    def putSend(self, in_name, in_data, in_type=None):
        if not self.comChek():
            return False
        while self.lock:
            time.sleep(0.001)
        self.lock = True
        delitem_temp = 0
        for delitem_loop in self.seit:
            if delitem_loop['name'] == in_name:
                delitem_loop['data'] = in_data
                if in_type is not None:
                    delitem_loop['type'] = in_type
                self.seit[delitem_temp] = delitem_loop
                self.lock = False
                return True
            delitem_temp = delitem_temp + 1
        self.lock = False
        return False

    # ------------------------------------------------
    #
    #              返回接受的某个变量的数据
    #
    #                   输入：变量名
    # ------------------------------------------------
    def getRecv(self, in_name):
        return self.getData(self.reit, in_name)

    # ------------------------------------------------
    #
    #              返回所有接受的变量名称
    #
    #       in_flag:True-输出类型，False-只输出名称
    # ------------------------------------------------
    def lstRecv(self, in_flag=False):
        if not self.comChek():
            return False
        return self.lstData(in_item=self.reit, in_flag=in_flag)

    # ------------------------------------------------
    #
    #              返回一个变量的数据实体（内部函数）
    #
    #                   输入：变量对象列表
    # ------------------------------------------------
    def getData(self, in_item, in_name):
        if not self.comChek():
            return False
        while self.lock:
            time.sleep(0.001)
        self.lock = True
        for delitem_loop in in_item:
            if delitem_loop['name'] == in_name:
                self.lock = False
                return delitem_loop
        self.lock = False
        return None

    # ------------------------------------------------
    #
    #                 清空系统缓冲区
    #
    # ------------------------------------------------
    def swpFree(self):
        if not self.comChek():
            return False
        while self.lock:
            time.sleep(0.001)
        self.lock = True
        self.buff = bytes()
        self.flag = 0
        self.lock = False
        return True

    # ------------------------------------------------
    #
    #             删除所有数据并重启串口
    #
    # ------------------------------------------------
    def Restart(self):
        self.Destroy()
        self.comLoad()
        self.comInit()
        if debug_flag:
            debugs.Debugs(debug_head, "SERIAL RESTART!", 4)

    # ------------------------------------------------
    #
    #             删除所有数据，关闭串口
    #
    # ------------------------------------------------
    def Destroy(self):
        if not self.comChek():
            return False
        self.init = False
        self.coms.close()
        self.coms = None
        self.buff = bytes()
        self.flag = 0
        self.sedt = {}
        self.seit = []
        self.redt = {}
        self.reit = []
        if debug_flag:
            debugs.Debugs(debug_head, "SERIAL DESTORY!", 4)

    # ------------------------------------------------
    #
    #                    初始化函数
    #
    # ------------------------------------------------
    def __init__(self,
                 in_port=0,
                 in_bbps=9600,
                 in_send=0,
                 in_dest=0,
                 in_nums=0,
                 in_time=0,
                 in_code="gbk",
                 ):
        self.port = str(in_port)  # 串口的端口号
        self.bbps = in_bbps  # 串口的波特率
        self.time = in_time  # 等待超时时间
        self.code = in_code  # 发送数据编码
        self.nums = in_nums  # 数据包的编号
        self.dest = in_dest  # 接收方标识号
        self.send = in_send  # 发送方标识号
        self.buff = bytes()  # 全局缓冲空间
        self.flag = 0  # 缓冲占用标识
        self.init = False  # 串口初始标识
        self.lock = False  # 串口占用标识
        self.coms = None  # 串口收发模块
        self.sedt = {}  # 待发送的数据
        self.seit = []  # 待发送的项目
        self.redt = {}  # 收到对方数据
        self.reit = []  # 收到对方项目
        self.comLoad()  # 加载串口参数
        self.comInit()  # 启动串口硬件
