#!/bin/perl
use strict;
use warnings;

my ($data, $csvsuffix) = @ARGV;

sub print_help_exit() {
    print "Usage: $0 <datafile> <csvsuffix>\n";
    exit 1;
}

if (not defined $data or not defined $csvsuffix) {
    print_help_exit();
}

# name -> [results]
my %table;

open (my $datafh, "<", "$data")
    or die "Could not open $data\n";

# read each line of data
while (my $line = <$datafh>) {
    if (my ($test_name, $impl) = $line =~ m/running (\w+) with ([\w\s]+)\s*.../) {
        my $res = open (my $csvfh, ">", "$test_name-$impl.$csvsuffix.csv");
        if (!$res) {
            print "Could not open $test_name-$csvsuffix.csv\n";
            last;   # continue
        }

        # write header
        print $csvfh "Matrix Size, Time(s)\n";

        # read every value 
        while (my ($n, $seconds, $nanoseconds) = 
            <$datafh> =~ m/[\s\w]*\w+\[n=\s*(\d+)\].* (\d+)\s*s\s*\+\s*(\d+)\s*ns/) {
            my $time = $seconds + $nanoseconds / 10e+9;

            printf $csvfh "%d,%lf\n", $n, $time;
        }

        close $csvfh;
    } else {
        print "Skipping line: '$line'\n";
    }
}

close $datafh;
