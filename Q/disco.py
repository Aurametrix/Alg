from disco.worker.task_io import chain_reader
from disco.core import Job, result_iterator

def map(line, params):
    for word in line.split("\n"):
        yield word, 1

def reduce(iter, params):
    from disco.util import kvgroup
    for word, counts in kvgroup(sorted(iter)):
        yield word, sum(counts)

if __name__ == '__main__':
    job = Job().run(input=["tag://data:batch"],
                    map_reader=chain_reader,
                    map=map,
                    reduce=reduce)
    for word, count in result_iterator(job.wait(show=True)):
        print(word, count)
