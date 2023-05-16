import warnings
warnings.filterwarnings("ignore")  # 忽略UserWarning兼容性警告
import tensorflow as tf
tf.compat.v1.disable_eager_execution()  # 可以在2.0中使用1.0的方法
tf.get_logger().setLevel('ERROR')  # 关闭因为版本问题出现的警告信息
import numpy as np


#####################  hyper parameters  ####################
LR_A = 0.0001    # learning rate for actor
LR_C = 0.0002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # 软更新  0' = TAU * 0 + (1 - TAU) * 0'
BATCH_SIZE = 32  # 随机抽样数
OUTPUT_GRAPH = False  # 是否生成日志---输出图

###############################  DDPG  ####################################

class DDPG(object):
    def __init__(self, s_dim, r_dim, b_dim, o_dim, r_bound, b_bound):
        self.memory_capacity = 10000  # 经验回放的容量为10000
        # dimension
        self.s_dim = s_dim  # 每个边缘服务器的可用资源10+每个连接(ES与ES之间)的可用带宽100+每个用户的任务卸载位置user_num+每个用户的位置(坐标)user_num*2
        self.a_dim = r_dim + b_dim + o_dim
        self.r_dim = r_dim  # user_num
        self.b_dim = b_dim  # user_num
        self.o_dim = o_dim  # user_num * edge_num
        # self.a_bound
        self.r_bound = r_bound  # 63000000.0
        self.b_bound = b_bound  # 1000000000.0
        '''
        placeholder()函数是在神经网络构建graph的时候的占位符，此时并没有把要输入的数据传入模型，
        它只会分配必要的内存。等建立session，在会话中，运行模型的时候通过feed_dict()函数向占位符喂入数据。
        '''
        # S, S_, R
        self.S = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, s_dim], 's')
        self.S_ = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, s_dim], 's_')
        self.R = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, 1], 'r')
        # memory   (S,A,R,S_)
        self.memory = np.zeros((self.memory_capacity, s_dim * 2 + self.a_dim + 1), dtype=np.float32)  # s_dim + a_dim + r + s_dim
        self.pointer = 0  # 总经验组数

        '''
        tensorflow的运行机制属于“定义”与“运行”相分离，tensorflow定义的内容都在“图”这个容器中完成，
        1、一个“图”代表一个计算任务
        2、在模型运行的环节中，“图”在会话(session)里被启动在tensorflow中定义的时候，
        其实就只是定义了图，图是静态的，在定义完成之后是不会运行的
        想让进行 图 中的节点操作，就需要使用运行函数 tf.Session.run 才能开始运行
        所有关于图中变量的赋值和计算都要通过tf.Session的run来进行。
        
        Tensorflow的设计理念称之为计算流图，在编写程序时，首先构筑整个系统的graph，
        代码并不会直接生效，graph为静态的，类似于docker中的镜像。
        然后，在实际的运行时，启动一个session，程序才会真正的运行。

        构建完整Graph后，利用 Session会话实例.run 将训练/测试数据注入到Graph中，驱动任务的实际运行
        '''
        # session
        self.sess = tf.compat.v1.Session()

        # 定义输入和输出
        self.a = self._build_a(self.S,)  # 用当前actor得到a  即  a = [resource  bandwidth  offloading]
        q = self._build_c(self.S, self.a, )  # 用当前critic得到q
        """
        get_collection()用于获取指定名称(这里是GraphKeys.TRAINABLE_VARIABLES)的集合,该函数返回一个列表，包含了指定名称的所有集合元素。
        """
        # 得到Actor和Critic中的参数
        a_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        """
        定义了一个指数移动平均（Exponential Moving Average，EMA）对象，
        通过设置衰减因子decay=1-TAU来控制EMA对历史数据的影响程度
        """
        ema = tf.compat.v1.train.ExponentialMovingAverage(decay=1 - TAU)  # 1.准备软更新    TAU    (1 - TAU)
        """
        该函数在获取权重参数时，会先利用EMA对象对历史参数进行加权平均，再返回平均后的参数值。
        这样做的好处是可以减少目标网络参数的抖动，提高模型的稳定性。
        """
        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))  # 对getter函数获取的某些数据进行指数移动平均操作

        # 调用ema.apply方法来实现对目标网络参数的更新
        # target_update为操作列表 它没有存放更新后的参数
        target_update = [ema.apply(a_params), ema.apply(c_params)]      # 2.进行软更新  0' = TAU * 0 + (1 - TAU) * 0'
        """
        具体来说，就是将主网络的参数传递给EMA对象，让EMA对象对其进行更新。最后，在构建目标网络时，
        我们通过设置custom_getter参数为ema_getter，即可让目标网络使用EMA更新后的参数。
        """
        a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)  # 3.custom_getter=ema_getter：目标actor使用软更新后的参数
        q_ = self._build_c(self.S_, a_, reuse=True, custom_getter=ema_getter)
        # Actor learn()
        a_loss = - tf.compat.v1.reduce_mean(q)  # a_loss是Actor网络优化的目标函数
        self.atrain = tf.compat.v1.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=a_params)
        '''
        with作用：文件读取时可以自动处理异常并且关闭文件，最后清理对象释放内存
        
        先执行enter()方法
        再执行with语句
        最后执行exit()，方法内会自带当前对象的清理方法。
        '''
        # Critic learn()
        with tf.compat.v1.control_dependencies(target_update):  # 表示任何其他操作之前先执行 target_update 中的所有操作
            q_target = self.R + GAMMA * q_  # 得到目标q
            td_error = tf.compat.v1.losses.mean_squared_error(labels=q_target, predictions=q)  # TD误差：q_target - q
            self.ctrain = tf.compat.v1.train.AdamOptimizer(LR_C).minimize(td_error, var_list=c_params)
        # sess.run()对图中的节点（变量）进行赋值或计算
        self.sess.run(tf.compat.v1.global_variables_initializer())  # 将所有图的变量进行集体初始化

        if OUTPUT_GRAPH:
            tf.compat.v1.summary.FileWriter("logs/", self.sess.graph)

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        """
        0~10000中选取32个数字，可重复  eg. indices =[555,120,431,150,555,...]
        根据列表中的值分别取出对应的经验组 再从这32个经验组中取出 s a r s_
        然后进行一次训练操作  atrain和ctrain
        """
        indices = np.random.choice(self.memory_capacity, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        """
        self.sess.run() 是 TensorFlow 中执行一组操作的方法这里它接受两个参数
        第一个参数（self.atrain）指定要执行的 TensorFlow 训练操作
        第二个参数（{self.S: bs}）提供了占位符所需的输入值字典
        """
        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):  # 存储经验组
        """
        存储经验组 如果超过10000就按顺序进行覆盖操作
        """
        transition = np.hstack((s, a, [r], s_))  # 将变量s、a、r和s_沿着水平方向（即行方向）拼接成一个新的数组。
        index = self.pointer % self.memory_capacity  # 按顺序得到下标然后存储 如果pointer大于memory_capacity就开始进行覆盖操作
        self.memory[index, :] = transition  # index取值范围在 [0 , 9999]
        self.pointer += 1

    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        '''
        variable_scope函数生成的上下文管理器会创建一个命名空间，可以用来管理变量  
        属性reuse=True生成上下文管理器时，这个上下文管理器内所有的tf.get_variable会直接获取
        已经创建的变量，如果变量不存在，会报错；
        但是若reuse=False或者None时，tf.get_variable会创建新的变量，如果同名参数存在会报错。
        '''
        with tf.compat.v1.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            n_l = 50
            '''
            tf.compat.v1.layers.dense(
                inputs, units, activation=None, use_bias=True, kernel_initializer=None,
                bias_initializer=tf.zeros_initializer(), kernel_regularizer=None,
                bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                bias_constraint=None, trainable=True, name=None, reuse=None)
            
            设定trainable=False可以防止该变量被数据流图的GraphKeys.TRAINABLE_VARIABLES收集，
            这样我们就不会在训练的时候尝试更新它的值。
            其实优化器优化的默认变量列表是数据流图的GraphKeys.TRAINABLE_VARIABLES，
            如果把trainable设置为True的话，就会把该变量放置到该列表中；
            如果为False，就不会放置到列表中，在训练时就不会更新变量。
            
            reuse属性默认为None表示不重复使用相同名称的上一次的权重。
            '''
            net = tf.compat.v1.layers.dense(s, n_l, activation=tf.compat.v1.nn.relu, name='l1', trainable=trainable)
            # resource ( 0 - r_bound)
            layer_r0 = tf.compat.v1.layers.dense(net, n_l, activation=tf.compat.v1.nn.relu, name='r_0', trainable=trainable)
            layer_r1 = tf.compat.v1.layers.dense(layer_r0, n_l, activation=tf.compat.v1.nn.relu, name='r_1', trainable=trainable)
            layer_r2 = tf.compat.v1.layers.dense(layer_r1, n_l, activation=tf.compat.v1.nn.relu, name='r_2', trainable=trainable)
            layer_r3 = tf.compat.v1.layers.dense(layer_r2, n_l, activation=tf.compat.v1.nn.relu, name='r_3', trainable=trainable)
            layer_r4 = tf.compat.v1.layers.dense(layer_r3, self.r_dim, activation=tf.compat.v1.nn.relu, name='r_4', trainable=trainable)

            # bandwidth ( 0 - b_bound)
            layer_b0 = tf.compat.v1.layers.dense(net, n_l, activation=tf.compat.v1.nn.relu, name='b_0', trainable=trainable)
            layer_b1 = tf.compat.v1.layers.dense(layer_b0, n_l, activation=tf.compat.v1.nn.relu, name='b_1', trainable=trainable)
            layer_b2 = tf.compat.v1.layers.dense(layer_b1, n_l, activation=tf.compat.v1.nn.relu, name='b_2', trainable=trainable)
            layer_b3 = tf.compat.v1.layers.dense(layer_b2, n_l, activation=tf.compat.v1.nn.relu, name='b_3', trainable=trainable)
            layer_b4 = tf.compat.v1.layers.dense(layer_b3, self.b_dim, activation=tf.compat.v1.nn.relu, name='b_4', trainable=trainable)

            # offloading (probability: 0 - 1)
            # layer
            # [['layer00', 'layer01', 'layer02', 'layer03'], ['layer10', 'layer11', 'layer12', 'layer13'],...]
            layer = [["layer"+str(user_id) + str(layer) for layer in range(4)] for user_id in range(self.r_dim)]
            # name
            name = [["layer"+str(user_id) + str(layer) for layer in range(4)] for user_id in range(self.r_dim)]
            # user
            # ['user0', 'user1', 'user2',...]
            user = ["user"+str(user_id) for user_id in range(self.r_dim)]
            # softmax
            # ['softmax0', 'softmax1', 'softmax2',...]
            softmax = ["softmax"+str(user_id) for user_id in range(self.r_dim)]
            for user_id in range(self.r_dim):  # 关于offloading有十套神经网络层  每一套得出的结果都按照顺序成为用户的卸载选择
                layer[user_id][0] = tf.compat.v1.layers.dense(net, n_l, activation=tf.compat.v1.nn.relu, name=name[user_id][0], trainable=trainable)
                layer[user_id][1] = tf.compat.v1.layers.dense(layer[user_id][0], n_l, activation=tf.compat.v1.nn.relu, name=name[user_id][1], trainable=trainable)
                layer[user_id][2] = tf.compat.v1.layers.dense(layer[user_id][1], n_l, activation=tf.compat.v1.nn.relu, name=name[user_id][2], trainable=trainable)
                layer[user_id][3] = tf.compat.v1.layers.dense(layer[user_id][2], (self.o_dim/self.r_dim), activation=tf.compat.v1.nn.relu, name=name[user_id][3], trainable=trainable)
                user[user_id] = tf.compat.v1.nn.softmax(layer[user_id][3], name=softmax[user_id])
            # concate
            a = tf.compat.v1.concat([layer_r4, layer_b4], 1)
            for user_id in range(self.r_dim):
                a = tf.compat.v1.concat([a, user[user_id]], 1)  # 将layer_r4，layer_b4以及user中的每一个策略选择都连接到一起（12个结果合一）
            return a

    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        # Q value (0 - inf)
        with tf.compat.v1.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l = 50
            '''
            get_variable(name=新变量或者现有变量的名称(取决于reuse的值),
                            shape= ~ 的形状,
                            trainable=是否在训练时更新其中的参数)
            
            当variable_scope使用参数reuse=True生成上下文管理器时，
            这个上下文管理器内所有的tf.get_variable会直接获取已经创建的变量，如果变量不存在，会报错；
            但是若reuse=False或者None时，tf.get_variable会创建新的变量，如果同名参数存在会报错。
            '''
            w1_s = tf.compat.v1.get_variable('w1_s', [self.s_dim, n_l], trainable=trainable)
            w1_a = tf.compat.v1.get_variable('w1_a', [self.a_dim, n_l], trainable=trainable)
            b1 = tf.compat.v1.get_variable('b1', [1, n_l], trainable=trainable)
            net_1 = tf.compat.v1.nn.relu(tf.compat.v1.matmul(s, w1_s) + tf.compat.v1.matmul(a, w1_a) + b1)
            net_2 = tf.compat.v1.layers.dense(net_1, n_l, activation=tf.compat.v1.nn.relu, trainable=trainable)
            net_3 = tf.compat.v1.layers.dense(net_2, n_l, activation=tf.compat.v1.nn.relu, trainable=trainable)
            net_4 = tf.compat.v1.layers.dense(net_3, n_l, activation=tf.compat.v1.nn.relu, trainable=trainable)
            return tf.compat.v1.layers.dense(net_4, 1, activation=tf.compat.v1.nn.relu, trainable=trainable)  # Q(s,a)