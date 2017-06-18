from vania.fair_distributor import FairDistributor

targets = ['Team A', 'Team B', 'Team C']
objects = ['Front-end Development', 'Back-end Development', 'Testing', 'Documentation']
weights = [
    [1, 2, 3, 2],		# hours for Team A to complete each task
    [3, 1, 4, 2],		# hours for Team B to complete each task
    [3, 4, 1, 1]		# hours for Team C to complete each task
]

distributor = FairDistributor(targets, objects, weights)
print(distributor.distribute())
