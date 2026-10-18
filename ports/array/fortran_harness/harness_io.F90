! Raw-binary I/O shared by every per-routine harness driver.
! Arrays are read/written as flat stream files <name>.bin in the driver's cwd.
module harness_io
  implicit none
contains
  subroutine rd(name, a, n)
    character(*), intent(in) :: name
    integer, intent(in) :: n
    real, intent(out) :: a(n)
    integer :: u
    open(newunit=u, file=name//'.bin', access='stream', form='unformatted', status='old')
    read(u) a
    close(u)
  end subroutine rd
  subroutine wr(name, a, n)
    character(*), intent(in) :: name
    integer, intent(in) :: n
    real, intent(in) :: a(n)
    integer :: u
    open(newunit=u, file=name//'.bin', access='stream', form='unformatted', status='replace')
    write(u) a
    close(u)
  end subroutine wr
end module harness_io
