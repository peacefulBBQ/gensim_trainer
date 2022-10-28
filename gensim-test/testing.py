import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.set_xlabel("Number of documents")
ax.set_ylabel("bound")

ax.plot([1,2,4],[1,2,3])
plt.show()