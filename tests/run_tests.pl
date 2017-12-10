#!/bin/env perl
use strict;
use warnings;

my @tests = (
    ['copy', 'dsdot', 'rot'], 
    ['gbmv', 'trmv', 'trsm'], 
    ['gemm', 'hemm']
);

my $matched = 0;
my $level;
my ($start, $end) = (1, 1);
my $type = 0;

do {
    print "Enter level (between 1 and 3): ";
    if (eof()) {
        exit 0;
    } elsif (($level) = (<> =~ m/^(\d+)$/)) {
        if ($level <= 3 and $level >= 1) {
            $matched = 1;
        }
    }
} while (!$matched); 

$matched = 0;
do {
    my $len = @{$tests[$level-1]};
    print "Enter range of tests (1-$len): ";

    if (eof()) {
        exit 0;
    } elsif (($start,$end) = (<> =~ m/(\d+)-(\d+)/)) {
        if (not ($start < 1 or $end > @{$tests[$level-1]}
                or $start > $end)) {
            $matched = 1;
        }
    }
} while (!$matched);

$matched = 0;
do {
    print "Enter type of test [1=b2c,2=mkl,3=nvblas]: ";

    if (eof()) {
        exit 0;
    } elsif (($type) = (<> =~ m/([123])/)) {
        $matched = 1;
    }
} while (!$matched);

my $awk_prog = "/[A-Z]+\\[n=[0-9]+\\]/{print}";
for (my $i = $start-1; $i < @{$tests[$level-1]} and $i <= $end; $i++) {
    my $prog = $tests[$level-1][$i-1];
    my $typename = $type == 1 ? "blas2cuda" : 
                ($type == 2 ? "Intel MKL" : "NVBLAS");
    print "running $prog with $typename...\n";
    for (my $N = 2; $N < 2**14; $N *= 2) {
        if ($type == 1) {
            my $objtrack_file = "$prog.objtrack.n_$N";
            if (! -e $objtrack_file) {
                system "cd ../scripts && ./objtrack.sh libmkl_intel_lp64.so ../tests/c/$prog $N --no-print-results 1>/dev/null 2>/dev/null && mv $prog.objtrack ../tests/$objtrack_file";
            }
            system "cd ../scripts && ./blas2cuda.sh ../tests/$objtrack_file ../tests/c/$prog $N --no-print-results 2>/dev/null | awk '$awk_prog'";
        } elsif ($type == 2) {
            system "./c/$prog $N --no-print-results";
        } elsif ($type == 3) {
            system "cd c && env LD_PRELOAD=$ENV{'CUDA'}/lib64/libnvblas.so NVBLAS_CONFIG_FILE=nvblas.conf ./$prog $N --no-print-results 2>/dev/null | awk '$awk_prog'";
        }
    }
}

