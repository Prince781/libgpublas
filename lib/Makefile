SOURCES=$(wildcard *.c)
OBJDIR=obj
OBJECTS=$(wildcard $(OBJDIR)/*.o)
CFLAGS=-Wall -Werror -fPIC -ggdb3
LDFLAGS=-ldl
TESTSOURCES=$(wildcard tests/*.c)
TESTS=$(TESTSOURCES:%.c=%)

libobjtracker.so: $(SOURCES:%.c=$(OBJDIR)/%.o)
	$(CC) -shared $(CFLAGS) $(LDFLAGS) $^ -o $@

$(OBJDIR)/%.o: %.c
	@if [ ! -d $(dir $@) ]; then mkdir -p $(dir $@); fi
	$(CC) $(CFLAGS) -shared $(LDFLAGS) -c $^ -o $@

tests/%: tests/%.c libobjtracker.so
	$(CC) -L$(PWD) -Wl,-rpath=$(PWD) -Wall -Werror -ggdb3 -o $@ $< -lobjtracker

.PHONY: clean

clean:
	@rm -rf $(OBJDIR)
	@rm -f $(OBJECTS)
	@rm -f libobjtracker.so
	@rm -f $(TESTS)
