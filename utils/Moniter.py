"""
监听train是的累计reward
"""

class Monitor:
    """
    todo 支持多线程和单线程
         有序的获取一个episodes的reward序列
         保存模型，
         支持tensorboard等
    """

    def __init__():

        pass
        
def im_processor(im):
    assert im.shape[2] == 3
    im = cv2.cvtColor(cv2.resize(im, (80, 80)), cv2.COLOR_BGR2GRAY)

    return im
