!flux_genny_rev3.f08

!Title:			Flux Generator (genny)
!Engineer:		Tom Looby
!Date:			03/16/2018
!Description:	Makes a set of Fluxes ANSYS / TensorFlow
!Project:       NSTX-U Recovery

!=======================================================================
!            ***MAIN PROGRAM***
!=======================================================================
program flux_genny
use iso_fortran_env
implicit none

integer :: i, j, ntime, nspace, iter, maxiter, failcheck
integer(8) :: t1, t2, rate, cmax

real(real64), ALLOCATABLE :: flux(:,:), r0(:), q(:), alpha(:), beta(:)
real(real64) :: t, dt, PI, dx, r, w, f_strike, spec_arr(9,2)
real(real64) :: c1, c2, c3, c4, xp, xc, P, Bp, fx, lambda, S
real(real64) :: rmin, rmax, tilemin, tilemax, tile_len
real(real64) :: squig1, squig2, squig3, squig4, squig5
real(real64) :: squig6, squig7, squig8, squig9
!real(real64) :: 
character(len=200) :: outfile, dir, command, flux_file


!=======================================================================
!             Setup 
!=======================================================================
!Initialize Random Number Generator
call init_random_seed()

!MC iterator
iter = 0

! Directory where results will be saved
dir = '/home/mobile1/school/grad/masters/flux_input/fluxes/fluxes_test/'
command = "mkdir " // trim(dir)
print *, command
failcheck = SYSTEM(trim(command))
if (failcheck == 1) then
   print *, "Failed to create subdirectory"
   CALL EXIT(0)
end if

!=======================================================================
!             Variables, Parameters, Machine Specs
!=======================================================================
PI=4.D0*DATAN(1.D0)
iter = 0

! User defined simulation parameters
print *, "How many spatial slices?"
read (*,*) nspace
print *, "How many time steps?"
read (*,*) ntime
print *, "How many Monte Carlo Runs?"
read (*,*) maxiter

ALLOCATE(flux(ntime, nspace), r0(ntime), q(ntime))
ALLOCATE(alpha(ntime), beta(ntime))

!Tile Specs
tile_len = 0.15431
dx = tile_len/nspace
tilemin = 0.438
tilemax = tilemin + tile_len

!From Memo, r0 range:
rmin = .46
rmax = .575

! Create Array with min and max values for machine specs
!  for each parameter: [minval, maxval]
!  C1 -> C4 are Eich Model Parameters
!
!rows are machine specs
!Generally, array looks like this:
!
! for array (i,j)
! i                  j1        j2
! Row #   Spec      MinVal    MaxVal
! 1       Bp         0.2       0.6
! 2       P          0.5       4.9
! 3       fx         4.0      30.0
! 4       t          1.0       5.0
! 5       c1         0.1       0.3
! 6       c2         1.0       2.5
! 7       c3        -0.1       0.25
! 8       c4        -1.4      -0.5
! 9       f_strike   0.0      20.0

! Build min/max array
!Bp [T]
spec_arr(1,1) = 0.2
spec_arr(1,2) = 0.6
!P [MW]
spec_arr(2,1) = 0.5
spec_arr(2,2) = 4.9
!fx
spec_arr(3,1) = 4.0
spec_arr(3,2) = 30.0
!t [s]
spec_arr(4,1) = 1.0
spec_arr(4,2) = 5.0
!c1
spec_arr(5,1) = 0.1
spec_arr(5,2) = 0.3
!c2
spec_arr(6,1) = 1.0
spec_arr(6,2) = 2.5
!c3
spec_arr(7,1) = -0.1
spec_arr(7,2) = 0.25
!c4
spec_arr(8,1) = -1.4
spec_arr(8,2) = -0.5
!f_strike [Hz]
spec_arr(9,1) = 0.0
spec_arr(9,2) = 20.0

!=======================================================================
!             Monte Carlo Machine Parameters
!=======================================================================
! Use a Monte Carlo method to pick random standard deviates from
! uniform distribution between boundaries in spec_arr
! This loop is designed to be run for as long as the user desires
! Alternatively, the user may edit the "iteration checker" to break
! after a specific number of runs.

!This is for CPU clocking.  Grab initial time.
CALL SYSTEM_CLOCK(t1, rate, cmax)

do while (1.NE.0)
   iter = iter + 1
   
   call random_number(squig1)
   call random_number(squig2)
   call random_number(squig3)
   call random_number(squig4)
   call random_number(squig5)
   call random_number(squig6)
   call random_number(squig7)
   call random_number(squig8)
   call random_number(squig9)
   
   Bp = spec_arr(1,1) + squig1*(spec_arr(1,2) - spec_arr(1,1))
   P = spec_arr(2,1) + squig2*(spec_arr(2,2) - spec_arr(2,1))
   fx = spec_arr(3,1) + squig3*(spec_arr(3,2) - spec_arr(3,1))
   
   ! Choose t from dist or keep constant
   !t = spec_arr(4,1) + squig4*(spec_arr(4,2) - spec_arr(4,1))
   t = 5.0
   
   !Eich Model Parameters (See NSTX PFC Memo)
   c1 = spec_arr(5,1) + squig6*(spec_arr(5,2) - spec_arr(5,1))
   c2 = spec_arr(6,1) + squig7*(spec_arr(6,2) - spec_arr(6,1))
   c3 = spec_arr(7,1) + squig8*(spec_arr(7,2) - spec_arr(7,1))
   c4 = spec_arr(8,1) + squig9*(spec_arr(8,2) - spec_arr(8,1))

   ! Sweep strike point (also have to uncomment R0 stuff below)
   f_strike = spec_arr(9,1) + squig5*(spec_arr(9,1) - spec_arr(9,2))
   
   dt = t/real(ntime) 
   w = 2*PI*f_strike
   
   
   ! r0 can change with time
   do i=1,ntime
      ! Function for r0 goes here:
      !time varying sinusoid
   !   r0(i) = (tile_len/2.0)*sin(w*(i-1)*dt) + tilemin
      
      !constant
      r0(i) = tilemin + tile_len/2.0
      
      ! Bound min and max r0 per project requirements
   !   if (r0(i) > rmax) then
   !      r0(i) = rmax
   !   elseif (r0(i) < rmin) then
   !      r0(i) = rmin
   !   end if 
   end do
   
   
   !=======================================================================
   !               Solve for S, lambda, and qmax
   !=======================================================================
   lambda = c2*(P**c3)*(Bp**c4)*10**(-3.0) ! convert to meters
   S = lambda*c1
   xp = S*fx
   xc = lambda*fx
   
   !q can be calculated from integral:
   ! P = integral(q(r)*2*pi*r*dr), bounded by r0-xp to r0 + xc
   
   do i=1,ntime
      alpha(i) = r0(i) - xp 
      beta(i) = r0(i) + xc
   
      q(i) = (P/(2*PI)) * ( (1/(3*xp)*(r0(i)**3 - alpha(i)**3)) - &
                            (alpha(i)/(2*xc)*(r0(i)**2 - alpha(i)**2)) + &
                            (1/2*(beta(i)**2 - r0(i)**2)) + &
                            (r0(i)/(2*xc)*(beta(i)**2 - r0(i)**2)) - &
                            (1/(3*xc)*(beta(i)**3 - r0(i)**3)) )**(-1.0)
   end do
   
   !=======================================================================
   !               Build Flux Profile
   !=======================================================================
   
   ! Build profile across tile surface
   do i = 1,ntime
      do j = 1, nspace
         r = tilemin + (j-1)*dx
         if (r >= alpha(i) .AND. r < r0(i)) then
            flux(i,j) = q(i)/xp * (r - alpha(i))
         elseif (r >= r0(i) .AND. r < beta(i)) then
            flux(i,j) = q(i) + q(i)/xc * (r0(i) - r) 
         else
            flux(i,j) = 0
         end if
      end do 
   end do
   
   !=======================================================================
   !            Write to CSV
   !=======================================================================
   !Change filename based upon iteration number
   write(outfile, "(A13,I0.6)") "flux_profile_", iter
   flux_file = trim(dir) // trim(outfile) // ".txt"
   
   OPEN(UNIT=1,FILE=trim(flux_file),FORM="FORMATTED",STATUS="REPLACE",ACTION="WRITE")
   ! write basic parameters for this test
   WRITE(1,*) '#===================================================================='
   WRITE(1,*) '# Heat flux generated by fortran program'
   WRITE(1,*) '# Parameters Listed Below:'
   WRITE(1,*) '# c1 = ', c1
   WRITE(1,*) '# c2 = ', c2
   WRITE(1,*) '# c3 = ', c3
   WRITE(1,*) '# c4 = ', c4
   WRITE(1,*) '# Bp = ', Bp
   WRITE(1,*) '# P = ', P
   WRITE(1,*) '# fx = ', fx
   WRITE(1,*) '# t = ', t
   WRITE(1,*) '# R0 time varying'
   WRITE(1,*) '#  '
   WRITE(1,*) '# This yields the following results...:'
   WRITE(1,*) '# Lambda [m]: ', lambda
   WRITE(1,*) '# S: ', S
   WRITE(1,*) '# xp [m]: ', xp
   WRITE(1,*) '# xc [m]: ', xc
   WRITE(1,*) '#===================================================================='
   
   
   ! This uses an inner implied do loop to iterate over array without newline
   do i = 1,ntime
      WRITE(UNIT=1, FMT=*) ((i-1)*dt), ",", (flux(i,j)*10**6, ",", j=1,nspace-1), flux(i,nspace)*10**6
   end do
   CLOSE(UNIT=1)
   
   !=======================================================================
   !            Final Output
   !=======================================================================
   if (mod(iter,1000) == 0) then 
      print *, 'Iteration Number', iter   
      CALL SYSTEM_CLOCK(t2, rate, cmax)
      print *, "Elapsed Time [s]: ", real(t2-t1)/real(rate)
      print *, ' '
      !print *, 'Directory Size: ', dir_size
      print *, ' '
      print *, ' '
   end if

   !=====Iteration checker=====
   !Use this if you want to break out after n iterations
   !Otherwise leave commented out
   if (iter >= maxiter) then
      print *, 'Reached Iteration Maximum'
      exit
   end if
   !============================
end do   



print *, ' '
print *, ' '
print *, 'Program Exexuted Successfully'
CALL SYSTEM_CLOCK(t2, rate, cmax)
print *, "Total Elapsed Time [s]: ", real(t2-t1)/real(rate)
print *, "Number of flux profiles created: ", iter


CONTAINS
   !This subroutine generates random number seed based upon clock
   SUBROUTINE init_random_seed()
     INTEGER :: i, n, clock
     INTEGER, DIMENSION(:), ALLOCATABLE :: seed
     CALL RANDOM_SEED(size = n)
     ALLOCATE(seed(n))
     CALL SYSTEM_CLOCK(COUNT=clock)
     seed = clock + 37 * (/ (i - 1, i = 1, n) /)
     CALL RANDOM_SEED(PUT = seed)
     DEALLOCATE(seed)
   END SUBROUTINE
          
end program flux_genny
