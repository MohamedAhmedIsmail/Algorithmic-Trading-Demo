var endpoint = 'api/data1';
    $.ajax({
        method: "GET",
        url: endpoint,
        success: function (data) {
            new Chart(document.getElementById("line-chart-closePrice"), 
            {
                type: 'line',
                data: data,
                options: {
                    title: {
                        display: true,
                        text: 'Madinet Nasr'
                    }
                }
            });
        },
        error: function (error_data) {
            console.log("error");
            console.log(error_data);
        }
    })
 var endpoint = 'api/data2';
    $.ajax({
        method: "GET",
        url: endpoint,
        success: function (data) {
            new Chart(document.getElementById("line-chart-Regression"), 
            {
                type: 'line',
                data: data,
                options: {
                    title: {
                        display: true,
                        text: 'Madinet Nasr'
                    }
                }
            });
        },
        error: function (error_data) {
            console.log("error");
            console.log(error_data);
        }
    })
var endpoint = 'api/data3';
    $.ajax({
        method: "GET",
        url: endpoint,
        success: function (data) {
            new Chart(document.getElementById("line-chart-closePrice-suez"), 
            {
                type: 'line',
                data: data,
                options: {
                    title: {
                        display: true,
                        text: 'Suez Cement'
                    }
                }
            });
        },
        error: function (error_data) {
            console.log("error");
            console.log(error_data);
        }
    })
    var endpoint = 'api/data4';
    $.ajax({
        method: "GET",
        url: endpoint,
        success: function (data) {
            new Chart(document.getElementById("line-chart-Regression-suez"), 
            {
                type: 'line',
                data: data,
                options: {
                    title: {
                        display: true,
                        text: 'Suez Cement'
                    }
                }
            });
        },
        error: function (error_data) {
            console.log("error");
            console.log(error_data);
        }
    })