! Harness driver for mp_thompson (src/thompson.F) on independent columns,
! through module_mp_thompson in its public variant
! (tests/fortran.py public_module: every entity public, same code).
! params = [nt_c, dt, nk, ncol, eqtset].  The table files written by
! drv_thompson_init (qr_acr_qgV2.dat, ...) must be in the run directory.
! Each column is set up exactly as mp_gt_driver does (non-aerosol: nc1d =
! Nt_c/rho, nwfa1d/nifa1d constants, ppt/vt/prv_rev/prw_vcd zeroed).
! Inputs a(nk, ncol): qv qc qi qr qs qg ni nr t p w dz.
program drv_thompson_column
  use harness_io
  use module_mp_thompson
  implicit none
  real :: params(5), hgt(1,1,1), dt
  integer :: nk, ncol, eqt, i, k
  real, allocatable, dimension(:,:) :: qv, qc, qi, qr, qs, qg, ni, nr, nc, t, p, w, dz, &
       prv, prw, ppt
  real, allocatable, dimension(:,:,:) :: vt
  real, allocatable, dimension(:) :: qv1d, qc1d, qi1d, qr1d, qs1d, qg1d, ni1d, nr1d, nc1d, &
       nwfa1d, nifa1d, t1d, p1d, w1d, dz1d, rho
  double precision, allocatable :: prv_rev(:), prw_vcd(:)
  real, allocatable, dimension(:) :: vtik, vtrk, vtsk, vtgk, vtck
  real :: pptrain, pptsnow, pptgraul, pptice

  call rd('params', params, size(params))
  dt = params(2); nk = nint(params(3)); ncol = nint(params(4)); eqt = nint(params(5))
  hgt = 0.
  call thompson_init(hgt=hgt, dowr=.false., nt_c_in=params(1), &
       ids=1, ide=2, jds=1, jde=2, kds=1, kde=2, ims=1, ime=1, jms=1, jme=1, &
       kms=1, kme=1, its=1, ite=1, jts=1, jte=1, kts=1, kte=1)
  allocate(qv(nk,ncol), qc(nk,ncol), qi(nk,ncol), qr(nk,ncol), qs(nk,ncol), qg(nk,ncol), &
       ni(nk,ncol), nr(nk,ncol), nc(nk,ncol), t(nk,ncol), p(nk,ncol), w(nk,ncol), &
       dz(nk,ncol), prv(nk,ncol), prw(nk,ncol), ppt(4,ncol), vt(nk+1,5,ncol))
  allocate(qv1d(nk), qc1d(nk), qi1d(nk), qr1d(nk), qs1d(nk), qg1d(nk), ni1d(nk), nr1d(nk), &
       nc1d(nk), nwfa1d(nk), nifa1d(nk), t1d(nk), p1d(nk), w1d(nk), dz1d(nk), rho(nk), &
       prv_rev(nk), prw_vcd(nk), vtik(nk+1), vtrk(nk+1), vtsk(nk+1), vtgk(nk+1), vtck(nk+1))
  call rd('qv', qv, size(qv)); call rd('qc', qc, size(qc)); call rd('qi', qi, size(qi))
  call rd('qr', qr, size(qr)); call rd('qs', qs, size(qs)); call rd('qg', qg, size(qg))
  call rd('ni', ni, size(ni)); call rd('nr', nr, size(nr)); call rd('t', t, size(t))
  call rd('p', p, size(p)); call rd('w', w, size(w)); call rd('dz', dz, size(dz))
  do i = 1, ncol
    prv_rev = 0.0; prw_vcd = 0.0
    vtik = 0.0; vtrk = 0.0; vtsk = 0.0; vtgk = 0.0; vtck = 0.0
    pptrain = 0.; pptsnow = 0.; pptgraul = 0.; pptice = 0.
    do k = 1, nk
      t1d(k) = t(k,i); p1d(k) = p(k,i); w1d(k) = w(k,i); dz1d(k) = dz(k,i)
      qv1d(k) = qv(k,i); qc1d(k) = qc(k,i); qi1d(k) = qi(k,i); qr1d(k) = qr(k,i)
      qs1d(k) = qs(k,i); qg1d(k) = qg(k,i); ni1d(k) = ni(k,i); nr1d(k) = nr(k,i)
      rho(k) = 0.622*p1d(k)/(R*t1d(k)*(qv1d(k)+0.622))
      nc1d(k) = Nt_c/rho(k)
      nwfa1d(k) = 11.1E6/rho(k)
      nifa1d(k) = naIN1*0.01/rho(k)
    enddo
    call mp_thompson(qv1d, qc1d, qi1d, qr1d, qs1d, qg1d, ni1d, nr1d, nc1d, nwfa1d, &
         nifa1d, t1d, p1d, w1d, dz1d, pptrain, pptsnow, pptgraul, pptice, prv_rev, &
         prw_vcd, vtik, vtrk, vtsk, vtgk, vtck, 1, nk, dt, i, 1, eqt)
    qv(:,i) = qv1d; qc(:,i) = qc1d; qi(:,i) = qi1d; qr(:,i) = qr1d; qs(:,i) = qs1d
    qg(:,i) = qg1d; ni(:,i) = ni1d; nr(:,i) = nr1d; nc(:,i) = nc1d; t(:,i) = t1d
    prv(:,i) = prv_rev; prw(:,i) = prw_vcd
    ppt(:,i) = (/ pptrain, pptsnow, pptgraul, pptice /)
    vt(:,1,i) = vtik; vt(:,2,i) = vtrk; vt(:,3,i) = vtsk; vt(:,4,i) = vtgk; vt(:,5,i) = vtck
  enddo
  call wr('qv', qv, size(qv)); call wr('qc', qc, size(qc)); call wr('qi', qi, size(qi))
  call wr('qr', qr, size(qr)); call wr('qs', qs, size(qs)); call wr('qg', qg, size(qg))
  call wr('ni', ni, size(ni)); call wr('nr', nr, size(nr)); call wr('nc', nc, size(nc))
  call wr('t', t, size(t)); call wr('prv_rev', prv, size(prv)); call wr('prw_vcd', prw, size(prw))
  call wr('ppt', ppt, size(ppt)); call wr('vt', vt, size(vt))
end program drv_thompson_column
