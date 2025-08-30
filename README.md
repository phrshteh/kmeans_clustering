
# How to compile and run the program?

# Parallel version
  - cd ~/kmeans_clustering/MPI
  - mpic++ -std=gnu++17 -O3 -o kmeans_api main.cpp
  - mpirun -np 14 ./kmeans_api

# Serial version
  - cd ~/kmeans_clustering/Serial
  - g++ -std=c++17 -O3 -o kmeans_serial main.cpp
  - ./kmeans_serial


# Developers

Fereshteh Mohammadi and Fatemeh Mohammadi are Computer Engineering students in University of Pavia.

