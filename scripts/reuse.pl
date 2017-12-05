#!/bin/env perl
# reuse.pl - track memory object reuse
use strict;
use warnings;

my ($track_file) = @ARGV;

if (not defined $track_file) {
    print "Usage: $0 <track file>\n";
    exit 1;
}

my $ext = "ltrace";
my ($progname) = $track_file =~ m/(.*)\.$ext$/;
if (not defined $progname) {
    print "track file must have '$ext' extension\n";
    exit 1;
}

# contains [reqsize, instruction pointer]
my @allocs = ();
my %addr_to_alloc;

open(my $track_file_h, "<", $track_file)
    or die "Could not open $track_file";

while (my $line = <$track_file_h>) {
    #   if (my ($reqsize, $address) = ($line =~ m/reqsize=[(\d+)] ip=[(0x[[:xdigit:]]+)]/)) {
    #   }
    # malloc?
    if (my ($ip, $reqsize, $addr) = ($line =~ m/\[0x([[:xdigit:]]+)\].*\W+malloc\((\d+)\).*= 0x([[:xdigit:]]+)/)) {
        # store the information
        # with the '[' syntax, we store a reference to an array,
        # instead of the array itself, as the value of a hash
        # must be a scalar
        $addr_to_alloc{hex($addr)} = [$reqsize, $ip];
    }
    # free?
    elsif (my ($addr2) = ($line =~ m/.*\W+free\(0x([[:xdigit:]]+)\).*= <void>/)) {
        # remove from our list
        delete $addr_to_alloc{hex($addr2)};
    }
    # blas?
    elsif (my ($addr3) = ($line =~ m/.*cblas_.*\(.*(0x([[:xdigit:]]+),?.*)+(\)|<unfinished \.\.\.>).*/)) {
        my @matches = $line =~ m/(0x([[:xdigit:]])+)/g;
        foreach my $addr4 (@matches) {
            if (defined $addr_to_alloc{hex($addr4)}) {
                push @allocs, $addr_to_alloc{hex($addr4)};
            }
        }
    }
    else {
        # something else? (then do nothing)
        print "Uncategorized line '$line'\n";
    }
}

close $track_file_h or die "Could not close $track_file";

open (my $output_fh, ">$progname.objtrack.1")
    or die "Failed to open $progname.objtrack for writing.";
# Now output allocs to file
foreach my $alloc (@allocs) {
    print $output_fh "reqsize=[$alloc->[0]] ip=[$alloc->[1]]\n"
}

close $output_fh or die "Could not close $progname.objtrack.1";
system("bash -c 'echo \"malloc\" | cat - <(sort -u $progname.objtrack.1) > $progname.objtrack'");
system("rm $progname.objtrack.1");
