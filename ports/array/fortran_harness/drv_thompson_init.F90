! Harness driver for thompson_init (src/thompson.F): constants, size bins and
! lookup tables, through module_mp_thompson_pub (thompson.F, all public).
! params = [nt_c].  Also writes the table files qr_acr_qgV2.dat,
! qr_acr_qsV2.dat and freezeH2O.dat (the format thompson_init reads back), so
! later drivers skip the expensive table computation.
program drv_thompson_init
  use harness_io
  use module_mp_thompson_pub
  implicit none
  real :: params(1), hgt(1,1,1), scal(60)
  integer :: n
  call rd('params', params, 1)
  hgt = 0.
  call thompson_init(hgt=hgt, dowr=.false., nt_c_in=params(1), &
       ids=1, ide=2, jds=1, jde=2, kds=1, kde=2, ims=1, ime=1, jms=1, jme=1, &
       kms=1, kme=1, its=1, ite=1, jts=1, jte=1, kts=1, kte=1)
  scal = 0.
  scal(1:11) = (/ Nt_c, mu_c, Sc3, D0i, xm0s, xm0g, oig1, oig2, obmi, obmr, ore1 /)
  scal(12:22) = (/ org1, org2, org3, oams, obms, ocms, oamg, obmg, ocmg, oge1, ogg1 /)
  scal(23:33) = (/ ogg2, ogg3, t1_qr_qc, t1_qr_qi, t2_qr_qi, t1_qg_qc, t1_qs_qc, &
       t1_qs_qi, t1_qr_ev, t2_qr_ev, t1_qs_sd /)
  scal(34:41) = (/ t2_qs_sd, t1_qs_me, t2_qs_me, t1_qg_sd, t2_qg_sd, t1_qg_me, &
       t2_qg_me, real(nic1) /)
  call wr('scalars', scal, size(scal))
  call wr('cce', cce, size(cce)); call wr('ccg', ccg, size(ccg))
  call wr('ocg1', ocg1, size(ocg1)); call wr('ocg2', ocg2, size(ocg2))
  call wr('cie', cie, size(cie)); call wr('cig', cig, size(cig))
  call wr('cre', cre, size(cre)); call wr('crg', crg, size(crg))
  call wr('cse', cse, size(cse)); call wr('csg', csg, size(csg))
  call wr('cge', cge, size(cge)); call wr('cgg', cgg, size(cgg))
  call wr('Dc', Dc, size(Dc)); call wr('dtc', dtc, size(dtc))
  call wr('Di', Di, size(Di)); call wr('dti', dti, size(dti))
  call wr('Dr', Dr, size(Dr)); call wr('dtr', dtr, size(dtr))
  call wr('Ds', Ds, size(Ds)); call wr('dts', dts, size(dts))
  call wr('Dg', Dg, size(Dg)); call wr('dtg', dtg, size(dtg))
  call wr('t_Nc', t_Nc, size(t_Nc))
  call wr('t_Efrw', t_Efrw, size(t_Efrw)); call wr('t_Efsw', t_Efsw, size(t_Efsw))
  call wr('tpc_wev', tpc_wev, size(tpc_wev)); call wr('tnc_wev', tnc_wev, size(tnc_wev))
  call wr('tps_iaus', tps_iaus, size(tps_iaus)); call wr('tni_iaus', tni_iaus, size(tni_iaus))
  call wr('tpi_ide', tpi_ide, size(tpi_ide))
  call wr('tcg_racg', tcg_racg, size(tcg_racg)); call wr('tmr_racg', tmr_racg, size(tmr_racg))
  call wr('tcr_gacr', tcr_gacr, size(tcr_gacr)); call wr('tmg_gacr', tmg_gacr, size(tmg_gacr))
  call wr('tnr_racg', tnr_racg, size(tnr_racg)); call wr('tnr_gacr', tnr_gacr, size(tnr_gacr))
  call wr('tcs_racs1', tcs_racs1, size(tcs_racs1)); call wr('tmr_racs1', tmr_racs1, size(tmr_racs1))
  call wr('tcs_racs2', tcs_racs2, size(tcs_racs2)); call wr('tmr_racs2', tmr_racs2, size(tmr_racs2))
  call wr('tcr_sacr1', tcr_sacr1, size(tcr_sacr1)); call wr('tms_sacr1', tms_sacr1, size(tms_sacr1))
  call wr('tcr_sacr2', tcr_sacr2, size(tcr_sacr2)); call wr('tms_sacr2', tms_sacr2, size(tms_sacr2))
  call wr('tnr_racs1', tnr_racs1, size(tnr_racs1)); call wr('tnr_racs2', tnr_racs2, size(tnr_racs2))
  call wr('tnr_sacr1', tnr_sacr1, size(tnr_sacr1)); call wr('tnr_sacr2', tnr_sacr2, size(tnr_sacr2))
  call wr('tpi_qrfz', tpi_qrfz, size(tpi_qrfz)); call wr('tni_qrfz', tni_qrfz, size(tni_qrfz))
  call wr('tpg_qrfz', tpg_qrfz, size(tpg_qrfz)); call wr('tnr_qrfz', tnr_qrfz, size(tnr_qrfz))
  call wr('tpi_qcfz', tpi_qcfz, size(tpi_qcfz)); call wr('tni_qcfz', tni_qcfz, size(tni_qcfz))
  open(63, file='qr_acr_qgV2.dat', form='unformatted')
  write(63) tcg_racg; write(63) tmr_racg; write(63) tcr_gacr
  write(63) tmg_gacr; write(63) tnr_racg; write(63) tnr_gacr
  close(63)
  open(63, file='qr_acr_qsV2.dat', form='unformatted')
  write(63) tcs_racs1; write(63) tmr_racs1; write(63) tcs_racs2; write(63) tmr_racs2
  write(63) tcr_sacr1; write(63) tms_sacr1; write(63) tcr_sacr2; write(63) tms_sacr2
  write(63) tnr_racs1; write(63) tnr_racs2; write(63) tnr_sacr1; write(63) tnr_sacr2
  close(63)
  open(63, file='freezeH2O.dat', form='unformatted')
  write(63) tpi_qrfz; write(63) tni_qrfz; write(63) tpg_qrfz; write(63) tnr_qrfz
  write(63) tpi_qcfz; write(63) tni_qcfz
  close(63)
  n = 0
end program drv_thompson_init
