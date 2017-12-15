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
my $N = 16;

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
    print "Enter size of input [N]: ";

    if (eof()) {
        exit 0;
    } elsif (($N) = (<> =~ m/(\d+)/)) {
        $matched = 1;
    }
} while (!$matched);

my $awk_prog = "/[A-Z]+\\[n=\\s+[0-9]+\\]/{print}";
my $diffprog = "diff";
if (-e "/bin/colordiff") {
    $diffprog = "/bin/colordiff";
}
#$diffprog .= " -y --suppress-common-lines";
$diffprog .= " -q";

for (my $i = $start-1; $i < @{$tests[$level-1]} and $i <= $end; $i++) {
    my $prog = $tests[$level-1][$i-1];
    for (my $type = 1; $type <= 3; ++$type) {
        my $typename = $type == 1 ? "blas2cuda" : 
		($type == 2 ? "Intel MKL" : "NVBLAS");
        print "running $prog with $typename...\n";
	if ($type == 1) {
	    my $objtrack_file = "$prog.objtrack.n_$N";
	    if (! -e $objtrack_file) {
		system "cd ../scripts && ./objtrack.sh libmkl_intel_lp64.so ../tests/c/$prog $N 1>/dev/null 2>/dev/null && mv $prog.objtrack ../tests/$objtrack_file";
	    }
	    system "cd ../scripts && ./blas2cuda_prof.sh ../tests/$objtrack_file ../tests/c/$prog $N 2>/dev/null | awk '$awk_prog'";
	    system "mv ./c/$prog.out $prog.b2c.out";
	} elsif ($type == 2) {
	    system "./c/$prog $N";
	    system "mv ./c/$prog.out $prog.mkl.out";
	} elsif ($type == 3) {
	    system "cd ../scripts && ./run_nvblas.sh ../tests/c/nvblas.conf ../tests/c/$prog $N 2>/dev/null | awk '$awk_prog'";
	    system "mv ./c/$prog.out $prog.nvb.out";
	}
    }

    print "Checking blas2cuda against MKL...\n";
    system "$diffprog $prog.b2c.out $prog.mkl.out";
    print "Checking NVBLAS against MKL...\n";
    system "$diffprog $prog.nvb.out $prog.mkl.out";
    print "Checking blas2cuda against NVBLAS...\n";
    system "$diffprog $prog.b2c.out $prog.nvb.out";
}

