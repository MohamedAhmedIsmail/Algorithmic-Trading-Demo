var app_companies = new Vue({
  delimiters: ["{[", "]}"],
  el: "#app_companies",
  data: {
    stock_data: undefined
  },
  methods: {
    getStock: function(event) {
      console.log(event.target.name);
      let stock_symbol = event.target.name;
      this.$http
        .get(`/stocks/api/stock/${stock_symbol}`)
        .then(Response => {
          this.stock_data = Response.data;
          console.log(this.stock_data);
          new Chart(document.getElementById("line-chart"), {
            type: "line",
            data: this.stock_data,
            options: {
              title: {
                display: true,
                text: event.target.textContent
              }
            }
          });
        })
        .catch(err => {
          console.log(err);
        });
    }
  }
});
