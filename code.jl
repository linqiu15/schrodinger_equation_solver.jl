using DifferentialEquations,BoundaryValueDiffEq,OrdinaryDiffEq;

μ_exp=0.140;mp_exp=0.938;mn_exp=0.939;hbarc=0.1973269804;

function exponential_V!(du,u,p,r;U0=1.4132604267565418,μ=μ_exp)
    U(r)=U0*exp(-μ*r/hbarc)
    μpn=mp_exp*mn_exp/(mp_exp+mn_exp)
    du[1]=u[2]
    du[2]=-(U(r)+2*μpn/hbarc^2*p)*u[1]
end

let 
f(x)=let u0=[0.0,x],rspan=(1e-10,80.0)    
    prob(e)=ODEProblem(exponential_V!,u0,rspan,e)
    solf(e)=solve(prob(e), Tsit5(), reltol=1e-8, abstol=1e-8)
    e_eigen=find_zero(e->solf(e).u[end][1],(-1.0,-0.00001))
    
    sol=solf(e_eigen);
    print("The eigen energy is $e_eigen\n")   
    
    xdata=sol.t
    ydata=hcat(sol.u...)[1,:]
    nydata=my_normalize(xdata,ydata)
    plot(xdata,nydata,label="ODE",ls=:dash)
end
    f(1e-3)
end