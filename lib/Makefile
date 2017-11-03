SOURCES=$(wildcard *.c)
OBJDIR=obj
OBJECTS=$(wildcard $(OBJDIR)/*.o)
CFLAGS=-Wall -Werror -fPIC -ggdb3
LDFLAGS=-ldl

libobjtracker.so: $(SOURCES:%.c=$(OBJDIR)/%.o)
	$(CC) -shared $(CFLAGS) $(LDFLAGS) $^ -o $@

$(OBJDIR)/%.o: %.c
	@if [ ! -d $(dir $@) ]; then mkdir -p $(dir $@); fi
	$(CC) $(CFLAGS) -shared $(LDFLAGS) -c $^ -o $@

.PHONY: clean

clean:
	@rm -rf $(OBJDIR)
	@rm -f $(OBJECTS)
	@rm -f libobjtracker.so
