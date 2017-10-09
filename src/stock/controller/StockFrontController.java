package stock.controller;

import java.io.IOException;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import stock.model.StockService;

public class StockFrontController extends HttpServlet {
	protected void service(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
		request.setCharacterEncoding("euc-kr");
		String command = request.getParameter("command");
		try {
			if (command.equals("stockSearch")) {
				getStock(request, response);
			}
		} catch (Exception s) {
			request.setAttribute("errorMsg", s.getMessage());
			request.getRequestDispatcher("showError.jsp").forward(request, response);
			s.printStackTrace();
		}
	}

	// 이름에 따른 주식정보 검색''''''''''''''''''''''''''''''''''''''''''''''''''''''
	public void getStock(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
		String url = "showError.jsp";
		try {
			request.setAttribute("result", StockService.getStock(request.getParameter("stockName")));
			url = "stock.jsp";
		} catch (Exception s) {
			request.setAttribute("errorMsg", s.getMessage());
		}
		request.getRequestDispatcher(url).forward(request, response);
	}
}
