cm1k_add_test(dynamics_tests SOURCES dyncore_step_tests.cpp)
cm1k_add_test(dynamics_options_tests SOURCES dyn_options_step_tests.cpp)
# testcase_tests runs too long for one ctest timeout in the bounds-checked
# Debug build, so it is registered as four tests over disjoint tag filters
# (quoted: ctest runs them through sh -c).
cm1k_add_test(testcase_tests SOURCES testcase_step_tests.cpp EXE_ARGS "'~[init]~[kessler]'")
EkatCreateUnitTestFromExec(testcase_tests_kessler testcase_tests EXE_ARGS "'~[init][kessler]'" LABELS cm1k)
EkatCreateUnitTestFromExec(testcase_tests_init testcase_tests EXE_ARGS "'[init]~[nosfc]'" LABELS cm1k)
EkatCreateUnitTestFromExec(testcase_tests_nosfc testcase_tests EXE_ARGS "'[nosfc]'" LABELS cm1k)
