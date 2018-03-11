tf.summary.sclar('이름',텐서(변수))
writer = tf.summay.FileWriter('디렉토리명',sess.graph) #파일 위치 
tf.summary.merge_all() # 위에 summary 할! 것들을 읽어드림
summary = sess.run(tf.summary.merge_all(),feed={}) 
writer.add_summary(summary,global_step = )






saver = tf.train.Saver(tf.global_variables())
ckpt = tf.train.get_checkpoint_state('체크포인트 디렉토리명')
tf.train_checkpoint_exist('경로(ckpt.model_checkpoint_path')
saver.restore(sess,chpt.model_checkpoint.path)
saver.save(세션,'경로',글로벌스탭)



