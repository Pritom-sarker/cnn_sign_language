import cv2
import My_Dl_lib as mdl
from PIL import Image
import tensorflow as tf
import numpy as np

def capture_image():
    cam = cv2.VideoCapture(0)


    cv2.namedWindow("test")

    img_counter = 0

    while True:
        ret, frame = cam.read()
        cv2.imshow("test", frame)
        if not ret:
            break
        k = cv2.waitKey(1)

        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            # SPACE pressed
            img_name = "test_image/sign_language_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()
    cv2.destroyAllWindows()
    return img_counter


#


if __name__=="__main__":
    #capture image
    #img_num=capture_image()

    #restore model
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name='input_x')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    logits = mdl.CNN(x, keep_prob)

    keep_probability = 0.7

    saver = tf.train.Saver()
    sess = tf.Session()
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    mdl._check_restore_parameters(sess, saver)


    for i in range(0,2):
        img_path = "test_image/sign_language_{}.png".format(i)
        img = Image.open(img_path).convert('L')
        img=img.resize((28,28))
        n_path='test_image/{}.png'.format(i)
        img.save(n_path)
        pix=mdl.get_image(n_path)
        log=sess.run(logits,feed_dict={x:pix,keep_prob:keep_probability})
        print(np.argmax(log))



