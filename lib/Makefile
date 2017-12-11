SOURCES=$(wildcard *.c) ../fortran-compat.c
OBJDIR=obj
OBJECTS=$(wildcard $(OBJDIR)/*.o)
CFLAGS=-Wall -Werror -fdiagnostics-color -fPIC -ggdb3 -std=gnu11 -DSTANDALONE -DTRACE_OUTPUT # -DRBTREE -DDEBUG_TRACKING -DTRACE_OUTPUT
LDFLAGS=-ldl -lunwind -lunwind-x86_64 -Wl,-init,obj_tracker_init,-fini,obj_tracker_fini
TESTSOURCES=$(wildcard tests/*.c)
TESTS=$(TESTSOURCES:%.c=%)

libobjtracker.so: $(SOURCES:%.c=$(OBJDIR)/%.o)
	$(CC) -shared $(CFLAGS) $(LDFLAGS) $^ -o $@

$(OBJDIR)/%.o: %.c ../cblas.h
	@if [ ! -d $(dir $@) ]; then mkdir -p $(dir $@); fi
	$(CC) $(CFLAGS) -shared $(LDFLAGS) -c $< -o $@

tests/%: tests/%.c libobjtracker.so
	$(CC) -L$(PWD) -Wl,-rpath=$(PWD) -Wall -Werror -ggdb3 -o $@ $< -lobjtracker

.PHONY: clean

clean:
	@rm -rf $(OBJDIR)
	@rm -f $(OBJECTS)
	@rm -f libobjtracker.so
	@rm -f $(TESTS)
