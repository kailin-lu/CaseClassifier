import sys 

import tensorflow as tf 
from tensorflow.contrib.tensorboard.plugins import projector

import numpy as np
from datetime import datetime
import random  

from layers import conv_layer

class TextCNN(): 
    def __init__(self, filter_sizes, filters, num_classes, pos_weight, 
                 reg_lambda, learning_rate, vocab_size, seq_length, embedding_size):
        self.filter_sizes = filter_sizes
        self.filters = filters 
        self.num_classes = num_classes 
        self.pos_weight = pos_weight 
        self.reg_lambda = reg_lambda
        self.learning_rate = learning_rate
        self.vocab_size = vocab_size 
        self.seq_length = seq_length
        self.embedding_size = embedding_size
        
    def _build_model(self, x, y, add_x, dropout_prob):
        tf.set_random_seed(0)
        # Embedding layer 
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.word_embeddings = tf.Variable(tf.random_uniform([self.vocab_size,
                                                                  self.embedding_size],
                                                                 -1.0, 1.0),
                                               name='word_embeddings')
            self.embedded_word_ids = tf.nn.embedding_lookup(self.word_embeddings, x)
            embedded_words_expanded = tf.expand_dims(self.embedded_word_ids, -1) 
        
        # Conv-Pool layers 
        self.pooled_outputs = [] 
        for i, filter_size in enumerate(self.filter_sizes):
            conv = conv_layer(embedded_words_expanded, kernel=filter_size,
                              channels_in=1,channels_out=self.filters, index=i)
            print('Conv {}'.format(i), conv.shape)
            # Pool across
            pooled = tf.nn.max_pool(conv, 
                            ksize=[1,x.shape[1]-filter_size+1, 1, 1],
                            strides=[1,1,1,1],
                            padding='VALID', 
                            name='pool{}'.format(i))
            print('Pooled {}'.format(i), pooled.shape)
            self.pooled_outputs.append(pooled)
            
        with tf.name_scope('fc'):
            total_filters = self.filters * len(self.filter_sizes)
            self.concat = tf.concat(self.pooled_outputs, 3)
            self.flattened = tf.reshape(self.concat, [-1,total_filters])
            print('Flatted: ', self.flattened.shape)
            
            self.concat = tf.concat((self.flattened, add_x), axis=1)  
            print('Concat:', self.concat.shape) 

            self.dropout = tf.nn.dropout(self.concat, dropout_prob, seed=0)
            print('Dropout:', self.dropout.shape)
            
            fc_logits = tf.layers.dense(self.dropout, self.num_classes)
            print('Logits: ', fc_logits.shape)
            
        with tf.name_scope('loss'):
            graph = tf.get_default_graph()
            l2_reg = tf.nn.l2_loss(graph.get_tensor_by_name('conv-weight-0:0'))
            l2_reg += tf.nn.l2_loss(graph.get_tensor_by_name('conv-weight-1:0'))
            try:
                l2_reg += tf.nn.l2_loss(graph.get_tensor_by_name('conv-weight-2:0'))
            except:
                pass
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, 
                                                                             logits=fc_logits)) 
            loss = loss + (self.reg_lambda*l2_reg)
        
        with tf.name_scope('prediction'):
            prediction = tf.argmax(fc_logits, 1)
            
        with tf.name_scope('accuracy'):
            correct = tf.equal(prediction, tf.argmax(y, 1)) 
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'), name='accuracy')
            
        return fc_logits, loss, prediction, accuracy


    def _step(self, loss):
        with tf.name_scope('step'): 
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            grads_and_vars = optimizer.compute_gradients(loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        return global_step, train_op


    def train(self, train_x, train_y, val_x, val_y,
              train_add, val_add, keep_prob, 
              geniss, metadata_path,
              batch_size=32, epochs=20):
        # Batch training data 
        batched_train_x = [train_x[i:i+batch_size] for i in range(0, len(train_x), batch_size)] 
        batched_train_y = [train_y[i:i+batch_size] for i in range(0, len(train_y), batch_size)] 
        batched_add_x = [train_add[i:i+batch_size] for i in range(0, len(train_add), batch_size)] 
        num_batches = len(batched_train_x) 
        range_batches = list(range(num_batches)) 
        print('Created {} batches of length {}'.format(num_batches, batch_size)) 
        
        tf.reset_default_graph()
        tf.set_random_seed(0)

        now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        root_logdir = 'tf_logs'
        logdir = '{}/run-{}'.format(root_logdir, now)
       
        with tf.name_scope('inputs'):
            x = tf.placeholder(shape=(None,self.seq_length), dtype=tf.int64, name='input_x')
            y = tf.placeholder(shape=(None,self.num_classes), dtype=tf.float32, name='input_y')
            add_x = tf.placeholder(shape=(None,7), dtype=tf.float32, name='add_x') 
            dropout_prob = tf.placeholder_with_default(1.0, shape=(), name='dropout_prob') 
            
        logits, loss, prediction, accuracy = self._build_model(x, y, add_x, dropout_prob)
        global_step, train_op = self._step(loss) 
        
        val_labels = np.argmax(val_y, 1) 
        
        embed_config = projector.ProjectorConfig() 
        embedding = embed_config.embeddings.add() 
        embedding.tensor_name = self.word_embeddings.name
        embedding.metadata_path = metadata_path
 
        tf.summary.scalar('loss', loss) 
        tf.summary.scalar('accuracy', accuracy) 
        merged = tf.summary.merge_all() 
        
        init = tf.global_variables_initializer() 
        saver = tf.train.Saver(max_to_keep=2)

        config = tf.ConfigProto()
        config.log_device_placement = True
        config.gpu_options.allow_growth = True
        
        with tf.Session(config=config) as sess:
            train_writer = tf.summary.FileWriter(logdir+'train/', sess.graph)
            val_writer = tf.summary.FileWriter(logdir+'val/', sess.graph)
            projector.visualize_embeddings(train_writer, embed_config)
            sess.run(init) 
              
            val_accuracies = [] 
            for epoch in range(epochs):
                train_acc = []
                random.shuffle(range_batches) 
                for batch in range_batches:
                    x_batch = batched_train_x[batch]
                    y_batch = batched_train_y[batch]
                    add_x_batch = batched_add_x[batch]
                    
                    _, step, err = sess.run([train_op,global_step,loss], 
                                            feed_dict={x: x_batch, 
                                                       y: y_batch, 
                                                       add_x:add_x_batch, 
                                                       dropout_prob: keep_prob}) 
                    pred = sess.run(prediction, feed_dict={x:x_batch, 
                                                           add_x:add_x_batch}) 
                    acc = sess.run(accuracy, feed_dict={x:x_batch, 
                                                        add_x:add_x_batch, 
                                                        y:y_batch}) 
                    train_acc.append(acc)
                    print('\r', 'Batch Accuracy:', acc, end='') 
                summary = sess.run(merged, feed_dict={x:x_batch, 
                                                      add_x:add_x_batch, 
                                                      y:y_batch}) 
                train_writer.add_summary(summary, epoch)
                saver.save(sess, 'final_cnn_model/{}/model.ckpt'.format(geniss), epoch)

                summary = sess.run(merged, feed_dict={x:val_x, 
                                                      add_x:val_add, 
                                                      y:val_y})
                val_writer.add_summary(summary, epoch)
                val_pred = sess.run(prediction, feed_dict={x:val_x, 
                                                           add_x:val_add})
                val_acc = sess.run(accuracy, feed_dict={x:val_x, 
                                                        add_x:val_add, 
                                                        y:val_y})
                val_accuracies.append(val_acc) 
                
                print('\r',end='') 
                print('Epoch {} Mean Train Acc {:2f} Validation Accuracy: {:2f}'.format(epoch,
                                                                                        np.mean(train_acc),
                                                                                        val_acc))
                # Early stopping condition 
                if len(val_accuracies) > 3 and sorted(val_accuracies[-3:], reverse=True) == val_accuracies[-3:]:
                    break
                    

            model_path = 'final_cnn_model/{}/geniss{}-acc{:4f}-lr{}'.format(geniss,
                                                                            geniss,
                                                                            val_acc,
                                                                            self.learning_rate)
            
            # Save model
            saver.save(sess, model_path)

            print('Saved Final Model. {}'.format(model_path))
            return val_pred


