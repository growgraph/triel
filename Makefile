RUNTEST=pytest test --linker-host 192.168.1.11

.PHONY: test
test:
	${RUNTEST} test