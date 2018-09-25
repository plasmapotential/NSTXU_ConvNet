#!/usr/bin/perl
# Date:         20180418
# Description:  Cleans Data for Tensorflow
# Engineer:     Tom Looby


use strict;
#se warnings;
use Path::Tiny qw(path);

#Set this variable for number of files to be cleaned
my $experN = 99;
my $i;
my $filename;
my $file;
my $data;

for ($i=1; $i<=$experN; $i++){

   $filename = sprintf("/home/workhorse/school/grad/masters/tensorflow/data_test_Cs_const1/TC_profile_%06i.txt", $i);
   $file = path($filename);

   #Read In Data, remove everything for TF, write new data
   $data = $file->slurp_utf8;
   $data =~ s/(\[|\]|C|')//g;
   $file->spew_utf8( $data );
}


print "\nCompleted.  Data Clean.\n\n"
