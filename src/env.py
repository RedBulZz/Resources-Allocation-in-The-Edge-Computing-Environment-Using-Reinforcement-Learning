import random
import numpy as np
import math
from render import Demo

#####################  hyper parameters  ####################
LOCATION = "KAIST"
USER_NUM = 30
EDGE_NUM = 10
LIMIT = 4  # 每个ES最多可以提供4个任务处理服务
MAX_EP_STEPS = 3000
TXT_NUM = 92
r_bound = 1e9 * 0.063  # ES的任务处理能力
b_bound = 1e9  # ES与ES之间的带宽

#####################  function  ####################
# B*math.log(1+SINR(self.user.loc, self.offloading_serv.loc), 2)/(8*time_scale)
def trans_rate(user_loc, edge_loc):
    B = 2e6
    P = 0.25
    d = np.sqrt(np.sum(np.square(user_loc[0] - edge_loc))) + 0.01
    h = 4.11 * math.pow(3e8 / (4 * math.pi * 915e6 * d), 2)
    N = 1e-10
    return B * math.log2(1 + P * h / N)


def BandwidthTable(edge_num):
    """
    将每个边缘服务器彼此之间的带宽设置为1e9（每两个只有一条带宽）
    任务可以在有限带宽（1e9 byte/sec）内从一个边缘服务器迁移到另一个边缘服务器

    :param edge_num: 边缘服务器数量  10个
    :return: 返回10*10的带宽矩阵
    """
    BandwidthTable = np.zeros((edge_num, edge_num))  # 10*10的矩阵
    for i in range(0, edge_num):
        for j in range(i+1, edge_num):
                BandwidthTable[i][j] = 1e9
    return BandwidthTable

def two_to_one(two_table):
    """
    降维 (10,10) -> (10 * 10) -> (100)  从矩阵降为列表

    :param two_table: 10 * 10的带宽矩阵 每两个ES之间都有一条带宽
    :return: 返回列表
    """
    one_table = two_table.flatten()  # 降维 (10,10) -> (10 * 10) -> (100)
    return one_table

def generate_state(two_table, U, E, x_min, y_min):
    """
    获得状态  是由每个边缘服务器的可用资源10、每个连接(ES与ES之间)的可用带宽100、
    每个用户的任务卸载位置user_num、每个用户的位置(坐标)user_num*2 组成的列表

    :param two_table: 10 * 10的带宽矩阵 每两个ES之间都有一条带宽
    :param U: 用户列表
    :param E: ES列表
    :param x_min: 所有文件中第二列里的最小值
    :param y_min: 所有文件中第三列里的最小值
    :return:
    """
    # initial
    one_table = two_to_one(two_table)
    S = np.zeros((len(E) + one_table.size + len(U) + len(U)*2))
    # transform
    count = 0
    # 每个边缘服务器的可用资源
    for edge in E:
        S[count] = edge.capability/(r_bound*10)  # 0.1
        count += 1
    # 每个连接(ES与ES之间)的可用带宽
    for i in range(len(one_table)):
        S[count] = one_table[i]/(b_bound*10)  # 0.1
        count += 1
    # 每个用户的任务卸载位置
    for user in U:
        S[count] = user.req.edge_id/100
        count += 1
    # 每个用户的位置(坐标) 每个user.loc对应两个元素 第一个元素是 此用户对应文件的第一行的第二列数据的变形 第二个元素是第一行的第三列数据的变形
    for user in U:
        S[count] = (user.loc[0][0] + abs(x_min))/1e5  # user.loc 此用户对应文件的第一行的第二第三列
        S[count+1] = (user.loc[0][1] + abs(y_min))/1e5
        count += 2
    return S

def generate_action(R, B, O):
    # resource
    a = np.zeros(USER_NUM + USER_NUM + EDGE_NUM * USER_NUM)
    a[:USER_NUM] = R / r_bound
    # bandwidth
    a[USER_NUM:USER_NUM + USER_NUM] = B / b_bound
    # offload
    base = USER_NUM + USER_NUM
    for user_id in range(USER_NUM):
        a[base + int(O[user_id])] = 1
        base += EDGE_NUM
    return a

def get_minimum():
    """
    整合所有文档中的第二列和第三列，并且找出最小值

    :return: 返回第二列的最小值，返回第三列的最小值
    """
    cal = np.zeros((1, 2))
    for data_num in range(TXT_NUM):  # 0 ~ 91
        data_name = str("%03d" % (data_num + 1))  # plus zero
        file_name = LOCATION + "_30sec_" + data_name + ".txt"
        file_path = "../data/" + LOCATION + "/" + file_name
        f = open(file_path, "r")
        f1 = f.readlines()
        # get line_num
        line_num = 0  # 记录每个文件的总行数
        for line in f1:
            line_num += 1
        # collect the data from the .txt
        data = np.zeros((line_num, 2))
        index = 0
        """
        line为f1列表中的每一项也就是文件里的每一行
        例如 0.0000000000000000e+000	 -3.8420858381879395e+002	 -4.6667833828169620e+001
        line.split对每一项/每一行进行拆分，并且返回一个列表
        例如 ['0.0000000000000000e+000', '-3.8420858381879395e+002', '-4.6667833828169620e+001']
        """
        for line in f1:
            data[index][0] = line.split()[1]  # x     取出第*行的第二个元素
            data[index][1] = line.split()[2]  # y     取出第*行的第三个元素
            index += 1
        # put data into the cal
        cal = np.vstack((cal, data))  # 将data矩阵放入cal矩阵中  循环最后的结果是 所有文档中的第二列和第三列都被加到cal中
    return min(cal[:, 0]), min(cal[:, 1])  # 选出cal中第一列中最小值和第二列中的最小值

def proper_edge_loc(edge_num):
    """
    一个边缘服务器对应九个文件 堆叠这九个文件的第二三列 并且求出第二和第三列的平均值  一共有10个边缘服务器 循环十次

    :param edge_num:10个边缘服务器
    :return:将平均值保存在矩阵中，如[[第一个服务器对应的文件的第二列的平均值,第一个服务器对应的文件的第三列的平均值],[ , ],...,[ , ]]
    """
    # initial the e_l
    e_l = np.zeros((edge_num, 2))
    # calculate the mean of the data
    group_num = math.floor(TXT_NUM / edge_num)  # 92 / 10 = 9    一个边缘服务器九个文件
    edge_id = 0
    for base in range(0, group_num*edge_num, group_num):  # base = 0  9  18  27  36  45  54  63  72  81
        for data_num in range(base, base + group_num):  # data_num = 0 ~ 89
            data_name = str("%03d" % (data_num + 1))  # plus zero
            file_name = LOCATION + "_30sec_" + data_name + ".txt"
            file_path = "../data/" + LOCATION + "/" + file_name
            f = open(file_path, "r")
            f1 = f.readlines()
            # get line_num and initial data
            line_num = 0
            for line in f1:
                line_num += 1
            data = np.zeros((line_num, 2))
            # collect the data from the .txt
            index = 0
            for line in f1:  # 取到这个文件中的所有第二列和第三列
                data[index][0] = line.split()[1]  # x
                data[index][1] = line.split()[2]  # y
                index += 1
            # stack the collected data
            if data_num % group_num == 0:  # cal中只堆叠同一个设备的文件（1个设备9个文件） 每过一个边缘服务器就刷新一下cal
                cal = data  # 用data来初始化cal
            else:
                cal = np.vstack((cal, data))  # 堆叠此边缘服务器的 9 个文件
        e_l[edge_id] = np.mean(cal, axis=0)  # 求这个边缘服务器对应的9个文件的（堆叠后的）第二第三列的平均值
        edge_id += 1
    return e_l

#############################UE###########################
class UE():
    def __init__(self, user_id, data_num):
        self.user_id = user_id  # number of the user
        self.loc = np.zeros((1, 2))
        self.num_step = 0  # the number of step

        # calculate num_step and define self.mob
        data_num = str("%03d" % (data_num + 1))  # plus zero    data_num + 1 是因为 data_num 的范围是 0~91 文件序号是 1~92
        file_name = LOCATION + "_30sec_" + data_num + ".txt"
        file_path = "../data/" + LOCATION + "/" + file_name
        f = open(file_path, "r")
        f1 = f.readlines()  # readlines()返回一个列表，将文件中的每一行作为一个列表元素。
        line_num = 0  # 记录每个文件的总行数           ！！！将data改为了line_num！！！
        for line in f1:  # 循环f1中的每一项，也就是文件里的每一行
            line_num += 1
        self.num_step = line_num * 30  # 每一行 30 步
        self.mob = np.zeros((self.num_step, 2))
        '''  
        mon = [[0. 0.]
               ...
               [0. 0.]]  
        self.num_step行   2列
        '''
        # write data to self.mob
        now_sec = 0
        for line in f1:
            for sec in range(30):  # 每一行 30 步 既 30 次循环
                self.mob[now_sec + sec][0] = line.split()[1]  # x
                self.mob[now_sec + sec][1] = line.split()[2]  # y
            now_sec += 30  # mob矩阵中30行对应文件中的一行   mob中的这30行每一行都一样  都是文件中对应那一行的第二第三列
        self.loc[0] = self.mob[0]  # 此用户对应的文件的第一行的第二第三列，将它作为此用户的坐标位置

    def generate_request(self, edge_id):
        self.req = Request(self.user_id, edge_id)

    def request_update(self):
        # default request.state == 5 means disconnection ,6 means migration
        if self.req.state == 5:
            self.req.timer += 1  # ？？？？？？ 如果未连接   timer加一
        else:
            self.req.timer = 0  # ？？？？？？  如果不为5就是正常连接  timer清零
            if self.req.state == 0:
                self.req.state = 1  # 开始将任务卸载到边缘服务器  传输
                self.req.u2e_size = self.req.tasktype.req_u2e_size  # 得到此用户传输到ES的数据量大小 270000
                self.req.u2e_size -= trans_rate(self.loc, self.req.edge_loc)  # 传输
            elif self.req.state == 1:
                if self.req.u2e_size > 0:  # 如果此用户正在将任务卸载到ES上但是任务数据还没卸载完
                    self.req.u2e_size -= trans_rate(self.loc, self.req.edge_loc)
                else:
                    self.req.state = 2  # 请求任务正在去往边缘服务器的路上 (2.7 * 1e4 字节)  处理
                    self.req.process_size = self.req.tasktype.process_loading  # 此用户对应ES能处理的总数据量 270000*4
                    self.req.process_size -= self.req.resource  # 数据量 270000*4 - 此用户所需要的资源量
            elif self.req.state == 2:
                if self.req.process_size > 0:  # 如果此用户对应ES还能处理
                    self.req.process_size -= self.req.resource
                else:
                    self.req.state = 3  # 请求任务已处理 (1.08 * 1e6 字节)   向回传输
                    self.req.e2u_size = self.req.tasktype.req_e2u_size  # ES处理完数据后得到的结果的大小
                    self.req.e2u_size -= 10000  # value is small,so simplify   传输
            else:
                if self.req.e2u_size > 0:  # 如果ES的结果没有传输完成
                    self.req.e2u_size -= 10000  # B*math.log(1+SINR(self.user.loc, self.offloading_serv.loc), 2)/(8*time_scale)
                else:
                    self.req.state = 4  # 请求任务正在返回移动用户的路上 （96 字节）  完成request！！！！！

    def mobility_update(self, time):  # t: second
        if time < len(self.mob[:, 0]):
            self.loc[0] = self.mob[time]  # 更新位置：以mob的第30行来作为此用户的位置
        else:
            self.loc[0][0] = np.inf
            self.loc[0][1] = np.inf


class Request():
    def __init__(self, user_id, edge_id):
        # id
        self.user_id = user_id
        self.edge_id = edge_id
        self.edge_loc = 0
        # state
        self.state = 5     # 5: not connect
        self.pre_state = 5
        # transmission size
        self.u2e_size = 0
        self.process_size = 0
        self.e2u_size = 0
        # edge info
        self.resource = 0
        self.mig_size = 0
        # tasktype

        self.tasktype = TaskType()
        self.last_offlaoding = 0
        # timer
        self.timer = 0


class TaskType():
    def __init__(self):
        ##Objection detection: VOC SSD300
        # transmission
        self.req_u2e_size = 300 * 300 * 3 * 1
        self.process_loading = 300 * 300 * 3 * 4
        self.req_e2u_size = 4 * 4 + 20 * 4
        # migration
        self.migration_size = 2e9

    def task_inf(self):
        return "req_u2e_size:" + str(self.req_u2e_size) + "\nprocess_loading:" + str(self.process_loading) + "\nreq_e2u_size:" + str(self.req_e2u_size)


#############################EdgeServer###################


class EdgeServer():
    def __init__(self, edge_id, loc):
        self.edge_id = edge_id  # edge server number 0 ~ 9
        self.loc = loc  # 此边缘服务器对应的9个文件的第二第三列平均值，将平均值作为此ES的坐标位置
        self.capability = 1e9 * 0.063  # 此ES的处理能力 6.3 * 1e7 byte/sec
        self.user_group = []
        self.limit = LIMIT
        self.connection_num = 0

    def maintain_request(self, R, U):
        """
        将应该装进对应的ES用户列表但没有装进的用户，并且在对应ES还有处理能力的时候以及没有超过4个任务的限制的时候
        ，就把用户装入对应的ES用户列表

        :param R: 更新过的 R ，每个用户需要消耗的资源  ，来自 a = [R B O]
        :param U: 用户列表
        """
        for user in U:
            # the number of the connection user
            self.connection_num = 0
            for user_id in self.user_group:  # 统计此ES的用户列表中req.state != 6的用户数目
                if U[user_id].req.state != 6:  # req.state != 6 已经开始迁移了
                    self.connection_num += 1
            # maintain the request
            # 如果此用户卸载对应的ES就是此ES and ES的处理能力 - 此用户需要消耗的资源 > 0
            if user.req.edge_id == self.edge_id and self.capability - R[user.user_id] > 0:
                # maintain the preliminary connection
                # 此用户不在此ES的用户列表中 and 此ES的用户列表中req.state != 6的用户数目 +1 < 4
                if user.req.user_id not in self.user_group and self.connection_num+1 <= self.limit:
                    # first time : do not belong to any edge(user_group)
                    self.user_group.append(user.user_id)  # add user.user_id to the user_group
                    user.req.state = 0  # prepare to connect and offloading
                    # notify the request
                    user.req.edge_id = self.edge_id
                    user.req.edge_loc = self.loc

                # dispatch the resource
                user.req.resource = R[user.user_id]
                self.capability -= R[user.user_id]

    def migration_update(self, O, B, table, U, E):
        # maintain the the migration
        for user_id in self.user_group:
            # prepare to migration
            if U[user_id].req.edge_id != O[user_id]:
                # initial
                ini_edge = int(U[user_id].req.edge_id)
                target_edge = int(O[user_id])
                if table[ini_edge][target_edge] - B[user_id] >= 0:
                    # on the way to migration, but offloading to another edge computer(step 1)
                    if U[user_id].req.state == 6 and target_edge != U[user_id].req.last_offlaoding:
                        # reduce the bandwidth
                        table[ini_edge][target_edge] -= B[user_id]
                        # start migration
                        U[user_id].req.mig_size = U[user_id].req.tasktype.migration_size
                        U[user_id].req.mig_size -= B[user_id]
                        #print("user", U[user_id].req.user_id, ":migration step 1")
                    # first try to migration(step 1)
                    elif U[user_id].req.state != 6:
                        table[ini_edge][target_edge] -= B[user_id]
                        # start migration
                        U[user_id].req.mig_size = U[user_id].req.tasktype.migration_size
                        U[user_id].req.mig_size -= B[user_id]
                        # store the pre state
                        U[user_id].req.pre_state = U[user_id].req.state
                        # on the way to migration, disconnect to the old edge
                        U[user_id].req.state = 6
                        #print("user", U[user_id].req.user_id, ":migration step 1")
                    elif U[user_id].req.state == 6 and target_edge == U[user_id].req.last_offlaoding:
                        # keep migration(step 2)
                        if U[user_id].req.mig_size > 0:
                            # reduce the bandwidth
                            table[ini_edge][target_edge] -= B[user_id]
                            U[user_id].req.mig_size -= B[user_id]
                            #print("user", U[user_id].req.user_id, ":migration step 2")
                        # end the migration(step 3)
                        else:
                            # the number of the connection user
                            target_connection_num = 0
                            for target_user_id in E[target_edge].user_group:
                                if U[target_user_id].req.state != 6:
                                    target_connection_num += 1
                            #print("user", U[user_id].req.user_id, ":migration step 3")
                            # change to another edge
                            if E[target_edge].capability - U[user_id].req.resource >= 0 and target_connection_num + 1 <= E[target_edge].limit:
                                # register in the new edge
                                E[target_edge].capability -= U[user_id].req.resource
                                E[target_edge].user_group.append(user_id)
                                self.user_group.remove(user_id)
                                # update the request
                                # id
                                U[user_id].req.edge_id = E[target_edge].edge_id
                                U[user_id].req.edge_loc = E[target_edge].loc
                                # release the pre-state, continue to transmission process
                                U[user_id].req.state = U[user_id].req.pre_state
                                #print("user", U[user_id].req.user_id, ":migration finish")
            #store pre_offloading
            U[user_id].req.last_offlaoding = int(O[user_id])

        return table

    #release the all resource
    def release(self):
        """
        释放所有资源 即self.capability = 1e9 * 0.063
        """
        self.capability = 1e9 * 0.063

#############################Policy#######################

class priority_policy():
    def generate_priority(self, U, E, priority):
        """
        获取卸载优先级矩阵，例如第一行为[[9,4,3,5,1,2,6,7,0,8],...]
        代表编号为0的用户优先卸载到编号为9的ES上，其次卸载到编号为4的ES上

        :param U: 用户列表
        :param E: ES列表
        :param priority: user_num * edge_num 的矩阵
        :return: 卸载优先级列表priority : user_num * edge_num 的矩阵
        """
        for user in U:
            dist = np.zeros(EDGE_NUM)  # [0,0,0,0,0,0,0,0,0,0]
            for edge in E:  # user.loc[0]此用户对应文件的第一行的第二三列    edge.loc此ES对应的九个文件的第二三列的平均值
                dist[edge.edge_id] = np.sqrt(np.sum(np.square(user.loc[0] - edge.loc)))  # 此用户和每个ES之间的直线距离
            dist_sort = np.sort(dist)  # 从小到大排列
            for index in range(EDGE_NUM):
                priority[user.user_id][index] = np.argwhere(dist == dist_sort[index])[0]  # 按照距离得到优先级矩阵
        return priority

    def indicate_edge(self, O, U, priority):
        """
        得出offloading列表 O, 列表中的元素代表用户把任务卸载到哪一个ES  因为受到ES只能接受4个任务（用户）
        所以列表中的元素最多会有四个相同的数字

        :param O: offloading列表  为user_num个元素的列表
        :param U: 用户列表
        :param priority: 卸载优先级列表  为user_num * edge_num 的矩阵
        :return: 返回offloading列表 O
        """
        edge_limit = np.ones((EDGE_NUM)) * LIMIT  # [4. 4. 4. 4. 4. 4. 4. 4. 4. 4.]
        for user in U:
            for index in range(EDGE_NUM):
                if edge_limit[int(priority[user.user_id][index])] - 1 >= 0:
                    edge_limit[int(priority[user.user_id][index])] -= 1
                    O[user.user_id] = priority[user.user_id][index]
                    break
        return O

    def resource_update(self, R, E ,U):
        for edge in E:
            # count the number of the connection user
            connect_num = 0
            for user_id in edge.user_group:
                if U[user_id].req.state != 5 and U[user_id].req.state != 6:
                    connect_num += 1
            # dispatch the resource to the connection user
            for user_id in edge.user_group:
                # no need to provide resource to the disconnecting users
                if U[user_id].req.state == 5 or U[user_id].req.state == 6:
                    R[user_id] = 0
                # provide resource to connecting users
                else:
                    R[user_id] = edge.capability/(connect_num+2)  # reserve the resource to those want to migration
        return R

    def bandwidth_update(self, O, table, B, U, E):
        for user in U:
            share_number = 1
            ini_edge = int(user.req.edge_id)
            target_edge = int(O[user.req.user_id])
            # no need to migrate
            if ini_edge == target_edge:
                B[user.req.user_id] = 0
            # provide bandwidth to migrate
            else:
                # share bandwidth with user from migration edge
                for user_id in E[target_edge].user_group:
                    if O[user_id] == ini_edge:
                        share_number += 1
                # share bandwidth with the user from the original edge to migration edge
                for ini_user_id in E[ini_edge].user_group:
                    if ini_user_id != user.req.user_id and O[ini_user_id] == target_edge:
                        share_number += 1
                # allocate the bandwidth
                B[user.req.user_id] = table[min(ini_edge, target_edge)][max(ini_edge, target_edge)] / (share_number+2)

        return B

#############################Env###########################

class Env():
    def __init__(self):
        self.step_30time = 30  # 30次向前传播用户移动一次
        self.time = 0  # ddpg_step_forward time
        self.edge_num = EDGE_NUM  # the number of servers
        self.user_num = USER_NUM  # the number of users
        # define environment object
        self.reward_all = []
        self.U = []
        self.fin_req_count = 0
        self.prev_count = 0
        self.rewards = 0
        self.R = np.zeros((self.user_num))  # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
        self.O = np.zeros((self.user_num))
        self.B = np.zeros((self.user_num))
        self.table = BandwidthTable(self.edge_num)
        self.priority = np.zeros((self.user_num, self.edge_num))  # user_num * edge_num 的矩阵
        self.E = []
        self.x_min, self.y_min = get_minimum()
        self.e_l = 0
        self.model = 0

    def get_inf(self):
        # s_dim
        self.reset()
        s = generate_state(self.table, self.U, self.E, self.x_min, self.y_min)
        s_dim = s.size

        # a_dim
        r_dim = len(self.U)  # user_num
        b_dim = len(self.U)
        o_dim = len(self.U) * self.edge_num

        # maximum resource
        r_bound = self.E[0].capability  # self.capability = 1e9 * 0.063

        # maximum bandwidth
        b_bound = self.table[0][1]
        b_bound = b_bound.astype(np.float32)

        # task size
        task = TaskType()
        task_inf = task.task_inf()

        return s_dim, r_dim, b_dim, o_dim, r_bound, b_bound, task_inf, LIMIT, LOCATION

    def reset(self):
        # reset time
        self.time = 0
        # reward
        self.reward_all = []
        # user
        self.U = []
        self.fin_req_count = 0
        self.prev_count = 0
        data_num = random.sample(list(range(TXT_NUM)), self.user_num)  # 从92个数据文档中随机抽取和用户数量相同的数据文档
        for i in range(self.user_num):
            new_user = UE(i, data_num[i])  # 一个用户对应一个随机的文件
            self.U.append(new_user)
        # Resource
        self.R = np.zeros((self.user_num))
        # Offlaoding
        self.O = np.zeros((self.user_num))
        # bandwidth
        self.B = np.zeros((self.user_num))
        # bandwidth table
        self.table = BandwidthTable(self.edge_num)
        # server
        self.E = []  # 装着10个边缘服务器对象
        e_l = proper_edge_loc(self.edge_num)
        for i in range(self.edge_num):
            new_e = EdgeServer(i, e_l[i, :])  # 创建10个边缘服务器对象
            self.E.append(new_e)
        # model
        self.model = priority_policy()  # 实例化 priority_policy类 为 model
        # initialize the request
        self.priority = self.model.generate_priority(self.U, self.E, self.priority)
        self.O = self.model.indicate_edge(self.O, self.U, self.priority)  # [9. 9. 9. 9. 0. 6. 2. 2. 2. 6. 2. 6. 0. 6. 1.]
        for user in self.U:
            user.generate_request(self.O[user.user_id])
        return generate_state(self.table, self.U, self.E, self.x_min, self.y_min)

    def ddpg_step_forward(self, a, r_dim, b_dim):
        # release the bandwidth
        self.table = BandwidthTable(self.edge_num)
        # release the resource
        for edge in self.E:
            edge.release()

        # update the policy every second
        # resource update
        self.R = a[:r_dim]  # 取出动作里R的部分
        # bandwidth update
        self.B = a[r_dim:r_dim + b_dim]  # 取出动作里B的部分
        # offloading update (O list update)
        base = r_dim + b_dim
        for user_id in range(self.user_num):
            prob_weights = a[base:base + self.edge_num]  # 取出动作中每个用户卸载到每个ES的概率
            # print("user", user_id, ":", prob_weights)
            action = np.random.choice(range(len(prob_weights)), p=prob_weights.ravel())  # select action w.r.t the actions prob
            base += self.edge_num
            self.O[user_id] = action  # 根据此用户卸载到每个ES的概率来选择最终的卸载结果

        # request update
        for user in self.U:
            # update the state of the request
            user.request_update()
            if user.req.timer >= 5:  # 如果此用户连续向前传播5次后 依然还是未连接(request.state == 5) 则此用户重新创建任务请求
                user.generate_request(self.O[user.user_id])  # offload according to the priority
            # 如果此用户完成了任务请求（在一回合中，此用户可以完成1000多次任务请求）
            if user.req.state == 4:
                # rewards
                self.fin_req_count += 1  # 所有用户完成任务请求数目
                user.req.state = 5  # request turn to "disconnect"
                self.E[int(user.req.edge_id)].user_group.remove(user.req.user_id)  # 若编号为1的用户完成了用户请求，就从它对应的ES的用户列表中删除编号1
                user.generate_request(self.O[user.user_id])  # offload according to the priority  此用户完成了任务请求，继续创建新的任务请求

        # edge update
        for edge in self.E:
            edge.maintain_request(self.R, self.U)
            self.table = edge.migration_update(self.O, self.B, self.table, self.U, self.E)

        # rewards
        self.rewards = self.fin_req_count - self.prev_count
        self.prev_count = self.fin_req_count

        # every user start to move
        if self.time % self.step_30time == 0:  # 向前传播30次，每个用户都走一步
            for user in self.U:
                user.mobility_update(self.time)

        # update time
        self.time += 1

        # return s_, r
        return generate_state(self.table, self.U, self.E, self.x_min, self.y_min), self.rewards

    def text_render(self):
        print("R:", self.R)
        print("B:", self.B)
        """
        base = USER_NUM +USER_NUM
        for user in range(len(self.U)):
            print("user", user, " offload probabilty:", a[base:base + self.edge_num])
            base += self.edge_num
        """
        print("O:", self.O)
        for user in self.U:
            print("user", user.user_id, "'s loc:\n", user.loc)
            print("request state:", user.req.state)
            print("edge serve:", user.req.edge_id)
        for edge in self.E:
            print("edge", edge.edge_id, "user_group:", edge.user_group)
        print("reward:", self.rewards)
        print("=====================update==============================")

    def initial_screen_demo(self):
        self.canvas = Demo(self.E, self.U, self.O, MAX_EP_STEPS)

    def screen_demo(self):
        self.canvas.draw(self.E, self.U, self.O)
