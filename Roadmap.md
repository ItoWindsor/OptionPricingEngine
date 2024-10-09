# Roadmap 

## Objective : 
- Create a robust pricing engine for derivatives 

### Sub Objectives :
- Be able to price multiple derivatives, the classical :  
  - european call/put option
  - european asian call/put option
  - european binary (up and in | up and out | down and in | down and out) call/put option 
  - European call/put Best of n-assets
  - 
  - american call/put option

- As well as the greeks :
  - delta 
  - gamma
  - vega

- Try different models: 
  - Black Scholes 
  - Merton 
  - Heston

- Through different methods : 
  - analytic (if available)
  - monte-carlo (should be good for every product but depends on the underlying model)
  - binomial tree (useful for derivatives with 1 underlying) -> jr/crr models