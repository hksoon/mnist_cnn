import logging
import subprocess

'''
def write_to_stderr_log(process):
    stderr= open("stderr.log", "w")
    proc_err = process.communicate()
    print(file=stderr)
    stderr.close()

def write_to_stdout_log(process):
    stdout = open("stdout.log", "w")
    proc_out = process.communicate()
    print(file=stdout)
    stdout.close()

def logger():
    logger = logging.getLogger('error_testing')
    hdlr = logging.FileHandler('error.log')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.WARNING)

    logger.error('We have a problem')
    logger.debug('debugging')
    logger.info('some info')

def log_main():
    #logger()
    proc = subprocess.Popen(['FastTree -nt test.fasta'], bufsize=512, stdin = None, stdout = subprocess.PIPE, stderr = subprocess.PIPE, shell=True)
    while proc.poll() is None:
        line = proc.stdout.readline()
        print("print : " + line)
#    write_to_stderr_log(proc)
#    write_to_stdout_log(proc)
'''
def myrun(cmd):
    """from http://blog.kagesenshi.org/2008/02/teeing-python-subprocesspopen-output.html
    """
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout = []
    while True:
        line = p.stdout.readline()
        stdout.append(line)
        print(line)
        if line == '' and p.poll() != None:
            break
    return ''.join(stdout)