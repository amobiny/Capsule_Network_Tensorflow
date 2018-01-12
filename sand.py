from sys import stdout
from time import sleep
for i in range(1,20):
    # stdout.write("\r%d" % i)
    print('\r{}'.format(i))
    # stdout.flush()
    sleep(1)