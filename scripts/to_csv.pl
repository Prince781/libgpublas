#!/bin/perl
use strict;
use warnings;
use POSIX ();

my ($data, $csvsuffix) = @ARGV;

sub print_help_exit() {
    print "Usage: $0 <datafile> <csvsuffix>\n";
    exit 1;
}

if (not defined $data or not defined $csvsuffix) {
    print_help_exit();
}

# n -> [results]
my %table;

my ($suffix) = $data =~ m/.+\.(.+)\..+/;

if (not defined $suffix) {
    die "Suffix not defined!";
}

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
        # print $csvfh "Matrix Size, Time(s)\n";

        if ($suffix ne "octave") {
            # read every value 
            while (my $line2 = <$datafh>) {
                if (my ($n, $seconds, $nanoseconds) = 
                    $line2 =~ m/[\s\w]*\w+\[n=\s*(\d+)\].* (\d+)\s*s\s*\+\s*(\d+)\s*ns/) {
                    my $time = $seconds + $nanoseconds / 10e+9;

                    printf $csvfh "%d,%lf\n", $n, $time;
                    push @{$table{$n}}, $time;
                } else {
                    last;
                }
            }
        } else {
            while (my $line2 = <$datafh>) {
                if (my ($n, $seconds, $gflops) =
                    $line2 =~ m/[\s\w]*\w+\[n=\s*(\d+)\].* elapsed=(\d+\.\d+)\s*s\s*GFLOPS=\s*(\d+.\d+)\s*/) {
                    my $time = $seconds;

                    printf $csvfh "%d,%lf\n", $n, $time;
                    push @{$table{$n}}, $time;
                } else {
                    last;
                }
            }
        }

        close $csvfh;
    } else {
        print "Skipping line: '$line'\n";
    }
}

print "LaTeX output for $suffix:\n";
foreach my $key (sort { $a <=> $b } keys %table) {
    print "\$n=$key\$";
    foreach my $val (@{$table{$key}}) {
        my $js = "js52";
        if (not -e "/bin/js52") {
            $js = "js38";
            if (not -e "/bin/js38") {
                $js = "";
            }
        }
        if ($js ne "") {
            printf " & \$%.5f\$", `$js -e "print ($val.toPrecision(3));"`;
        } else {
            printf " & \$%.5f\$", $val;
        }
    }
    print "\t\\\\ \n";
}

close $datafh;
