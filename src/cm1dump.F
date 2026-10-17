!-----------------------------------------------------------------------
!
!  cm1dump:  raw state dumps used as references by the C++/Kokkos and
!  array-backend ports (see cxx/README.md).
!
!  Enabled only when the environment variable CM1_DUMP_DIR is set.
!  CM1_DUMP_EVERY (default 1) sets how often (in steps) to dump.
!
!  Layout:  $CM1_DUMP_DIR/step_NNNNNN/<name>.bin holds the array as raw
!  native-endian reals (kind of the default real) in Fortran (column-major)
!  order, including halo points.  $CM1_DUMP_DIR/step_NNNNNN/index.txt has
!  one line per array:   name kind ndims lb1 ub1 [lb2 ub2 ...]
!  step_000000 is the initial state (grid, base state and prognostics).
!
!  Only serial (non-MPI) runs are supported.
!
!-----------------------------------------------------------------------

  MODULE cm1dump

  implicit none

  private
  public :: dump_enabled, dump_due, dump_begin, dump_end, dump1, dump3, dump4

  logical, save :: initialized = .false.
  logical, save :: enabled = .false.
  integer, save :: every = 1
  character(len=512), save :: basedir = ' '
  character(len=600), save :: curdir = ' '
  integer, parameter :: iunit_idx = 71, iunit_dat = 72

  CONTAINS

  subroutine dump_init()
    integer :: ln, st
    character(len=32) :: buf
    if( initialized ) return
    initialized = .true.
    call get_environment_variable('CM1_DUMP_DIR',basedir,ln,st)
    enabled = ( st.eq.0 .and. ln.gt.0 )
    call get_environment_variable('CM1_DUMP_EVERY',buf,ln,st)
    if( st.eq.0 .and. ln.gt.0 ) read(buf,*) every
    every = max(1,every)
  end subroutine dump_init

  logical function dump_enabled()
    call dump_init()
    dump_enabled = enabled
  end function dump_enabled

  logical function dump_due(nstep)
    integer, intent(in) :: nstep
    call dump_init()
    dump_due = enabled .and. ( mod(nstep,every).eq.0 )
  end function dump_due

  subroutine dump_begin(nstep,mtime)
    integer, intent(in) :: nstep
    double precision, intent(in) :: mtime
    write(curdir,'(a,a,i6.6)') trim(basedir),'/step_',nstep
    call execute_command_line('mkdir -p '//trim(curdir))
    open(unit=iunit_idx,file=trim(curdir)//'/index.txt',status='replace',form='formatted')
    write(iunit_idx,'(a,1x,i10,1x,es24.16)') '#step_time',nstep,mtime
  end subroutine dump_begin

  subroutine dump_end()
    close(unit=iunit_idx)
  end subroutine dump_end

  subroutine open_dat(name)
    character(len=*), intent(in) :: name
    open(unit=iunit_dat,file=trim(curdir)//'/'//trim(name)//'.bin',status='replace',  &
         form='unformatted',access='stream')
  end subroutine open_dat

  subroutine dump1(name,a,i1,i2)
    character(len=*), intent(in) :: name
    integer, intent(in) :: i1,i2
    real, intent(in), dimension(i1:i2) :: a
    call open_dat(name)
    write(iunit_dat) a
    close(unit=iunit_dat)
    write(iunit_idx,'(a,1x,i2,1x,i1,2(1x,i6))') trim(name),kind(a),1,i1,i2
  end subroutine dump1

  subroutine dump3(name,a,i1,i2,j1,j2,k1,k2)
    character(len=*), intent(in) :: name
    integer, intent(in) :: i1,i2,j1,j2,k1,k2
    real, intent(in), dimension(i1:i2,j1:j2,k1:k2) :: a
    call open_dat(name)
    write(iunit_dat) a
    close(unit=iunit_dat)
    write(iunit_idx,'(a,1x,i2,1x,i1,6(1x,i6))') trim(name),kind(a),3,i1,i2,j1,j2,k1,k2
  end subroutine dump3

  subroutine dump4(name,a,i1,i2,j1,j2,k1,k2,n1,n2)
    character(len=*), intent(in) :: name
    integer, intent(in) :: i1,i2,j1,j2,k1,k2,n1,n2
    real, intent(in), dimension(i1:i2,j1:j2,k1:k2,n1:n2) :: a
    call open_dat(name)
    write(iunit_dat) a
    close(unit=iunit_dat)
    write(iunit_idx,'(a,1x,i2,1x,i1,8(1x,i6))') trim(name),kind(a),4,i1,i2,j1,j2,k1,k2,n1,n2
  end subroutine dump4

  END MODULE cm1dump
