! dinterp module
! compute displacement interpolation (dinterp)
!
! Donsub Rim, Columbia U, (dr2965@columbia.edu) 2018-07-24
! see LICENSE

module dinterp

    contains

    subroutine cumsum(a,sa)
    ! compute cumulative sum
      implicit none
      save

      real(kind=8), dimension(:), intent(in) :: a
      real(kind=8), dimension(size(a,1)), intent(out) :: sa

      real(kind=8) :: b
      integer :: i,j,n

      n = size(a,1)
      b = 0d0

      do i=1,n
         b = b + a(i)
         sa(i) = b
      enddo
    end subroutine cumsum


    subroutine merge_monotone(xf,xg,F,G,tol,alph,beta,H,ln)
    ! merge two CDFs so that they share points on the y-axis

      implicit none
      save
      
      real(kind=8), dimension(:), intent(in) :: xf
      real(kind=8), dimension(:), intent(in) :: xg
      real(kind=8), dimension(:), intent(in) :: F
      real(kind=8), dimension(:), intent(in) :: G
      real(kind=8), intent(in) :: tol
      
      real(kind=8), dimension(size(xf,1)+size(xg,1)), intent(out) :: alph
      real(kind=8), dimension(size(xf,1)+size(xg,1)), intent(out) :: beta
      real(kind=8), dimension(size(xf,1)+size(xg,1)), intent(out) :: H
      integer, intent(out) :: ln
      real(kind=8), dimension(:), allocatable :: ins

      real(kind=8), dimension(2*size(F,1),3) :: y
      integer, dimension(:), allocatable :: jj
      integer, dimension(:), allocatable :: kk

      integer :: i,j,j0,j1,k,l1,l2,lt,m,nf,ng, alloc_error
      logical :: verbose

      real(kind=8) :: w1,w2,w3

      nf = size(F,1) 
      ng = size(G,1) 

      verbose = .true.

      lt = 0    ! no of skipped pts
      j = 0     ! start at 0, note index of F starts at 1
      k = 1     ! start at 1, note index of G starts at 1
      
      ! (1) F-interval has zero length
      do while ((j .lt. nf) .and. (k .lt. ng))
        if ( ((j .eq. 0) .and. (abs(F(1)-0d0) .le. tol)) &
        .or. ((j .gt. 0) .and. (j .lt. nf) .and. (abs(F(j+1)-F(j)) .le. tol)))&
           then
          
          l1 = 1
          do while (((j .eq. 0)  .and. (abs(F(j+l1+1)-0d0) .le. tol)) &
              .or.  ((j .gt. 0) .and. (abs(F(j+l1+1)-F(j)) .le. tol)))
            l1 = l1+1     ! length of constant interval to skip
          enddo
          
          if (j .eq. 0) then
            alph(j+k-lt) = xf(1)
            H(j+k-lt) = F(1)      ! F(1) = 0
          else if (j .gt. 0) then
            alph(j+k-lt) = xf(j)
            H(j+k-lt) = F(j)
          endif
          alph(j+k-lt+1) = xf(j+l1)
          H(j+k-lt+1) = F(j+l1)

          if   (((j .eq. 0) .and. (abs(G(k)-0d0) .le. tol))  &
           .or. ((j .gt. 0) .and. (abs(G(k)-F(j)) .le. tol))) then

            l2 = 0
            do while (((j .eq. 0) .and. (abs(G(k+l2+1)-0d0) .le. tol)) &
                .or.  ((j .gt. 0) .and. (abs(G(k+l2+1)-F(j)) .le. tol)))
              l2 = l2+1    ! length of constant interval to skip 
            enddo

            beta(j+k-lt) = xg(k)
            beta(j+k-lt+1) = xg(k+l2)
            k = k+l2+1    ! skip indices for k, pick next entry
            lt = lt+(l1-1)+l2
          else
            if (k .eq. 1) then
              w1 =   G(k) - F(j)
              w2 =   F(j) - 0d0 
              w3 =   G(k) - 0d0
              w1 = w1/w3
              w2 = w2/w3

              beta(j+k-lt) = xg(1) 
              beta(j+k-lt+1) = xg(1) 
            else if (k .gt. 1) then
              w1 =   G(k) - F(j)
              w2 =   F(j) - G(k-1) 
              w3 =   G(k) - G(k-1)
              w1 = w1/w3
              w2 = w2/w3
              
              beta(j+k-lt) = w1*xg(k-1) + w2*xg(k)
              beta(j+k-lt+1) = w1*xg(k-1) + w2*xg(k)
            endif
            lt = lt+(l1-1)-1
          endif   

          j = j+l1         ! skip indices for j
          ln = j+k-lt-1
        ! (2) F-interval has positive length
        else 
          ! G(k) belongs to the F(j)-interval
          if    (((j .eq. 0) .and. (G(k) .ge. 0d0) .and. (G(k) .lt. F(j+1))) &
            .or. ((j .gt. 0) .and. (j .lt. nf) &
                             .and. (G(k) .ge. F(j)) .and. (G(k) .lt. F(j+1))))&
            then

            if (j .eq. 0) then
              w1 = F(j+1) - G(k)
              w2 = G(k)
              w3 = F(j+1) - 0d0
              w1 = w1/w3
              w2 = w2/w3

              alph(j+k-lt) = xf(1)
            else 
              w1 = F(j+1) - G(k)
              w2 =   G(k) - F(j)
              w3 = F(j+1) - F(j)
              w1 = w1/w3
              w2 = w2/w3

              alph(j+k-lt) = w1*xf(j) + w2*xf(j+1)
            endif
            
            beta(j+k-lt) = xg(k)
            H(j+k-lt) = G(k)

            ! G-interval has zero length
            l2 = 0
            do while (abs(G(k+l2+1)-G(k)) .le. tol)
              l2 = l2+1
            enddo
            
            if (l2 .gt. 0) then
              alph(j+k-lt+1) = alph(j+k-lt)
              beta(j+k-lt+1) = xg(k+l2)
              H(j+k-lt+1) = G(k+l2)
              k = k+l2+1
              lt = lt+l2-1
            else
              k = k+1
            endif
          ! G(k) does not belong to the F(j)-interval
          else
            if (k .eq. 1) then
              w1 =   G(k) - F(j+1)
              w2 = F(j+1) - 0d0 
              w3 =   G(k) - 0d0
              w1 = w1/w3
              w2 = w2/w3

              beta(j+k-lt) = xg(1) 
            else if (k .gt. 1) then
              w1 =   G(k) - F(j+1)
              w2 = F(j+1) - G(k-1) 
              w3 =   G(k) - G(k-1)
              w1 = w1/w3
              w2 = w2/w3
              
              beta(j+k-lt) = w1*xg(k-1) + w2*xg(k)
            endif
              
            alph(j+k-lt) = xf(j+1)
            H(j+k-lt) = F(j+1)
            j = j+1
          endif   
          ln = j+k-lt-1
        endif
      enddo  
      ! kluging
      alph(ln) = xf(nf)
      beta(ln) = xg(ng)

    
    end subroutine merge_monotone



end module dinterp

