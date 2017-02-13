import tensorflow as tf

uniform_with_seed = tf.random_uniform([1], seed=1)
uniform_without_seed = tf.random_uniform([1])

print("First Run")
with tf.Session() as first_session:
    print("uniform with (seed = 1) = {}"\
          .format(first_session.run(uniform_with_seed)))
    print("uniform with (seed = 1) = {}" \
          .format(first_session.run(uniform_with_seed)))
    print("uniform without = {}" \
          .format(first_session.run(uniform_without_seed)))
    print("uniform without = {}" \
          .format(first_session.run(uniform_without_seed)))

print("Second Run")
with tf.Session() as second_session:
    print("uniform with (seed = 1) = {}"\
          .format(second_session.run(uniform_with_seed)))
    print("uniform with (seed = 1) = {}" \
          .format(second_session.run(uniform_with_seed)))
    print("uniform without = {}" \
          .format(second_session.run(uniform_without_seed)))
    print("uniform without = {}" \
          .format(second_session.run(uniform_without_seed)))

print("Note how the seeded values get same answer in the different sessions")
