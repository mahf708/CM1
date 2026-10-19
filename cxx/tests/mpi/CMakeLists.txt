# MPI domain decomposition (CM1K_ENABLE_MPI): run at 2 and 4 ranks.
#   mpi_tests_np2, mpi_tests_np4: decomposition and halo-exchange unit tests,
#   and the mpi_* reference cases, each rank compared bit for bit with the
#   MPI Fortran at the same rank count (cases/README.md, "MPI references").
cm1k_add_test(mpi_tests SOURCES mpi_comm_tests.cpp mpi_step_tests.cpp MPI_RANKS 2 4 2 LABELS mpi)
