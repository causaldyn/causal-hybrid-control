\\ PARI/GP: arbitrary-precision numeric gold reference for the scalar continuous ARE.
\\ Demonstrates precision beyond scipy's float64.  Run: gp -q validation/lqr_care.gp
\p 50
a = -1/10; b = 1; q = 1; r = 1;
pstab = (a + sqrt(a^2 + b^2*q/r)) * r / b^2;
print("scalar P (50 digits) = ", pstab);
print("scalar K (50 digits) = ", (b/r)*pstab);
quit
