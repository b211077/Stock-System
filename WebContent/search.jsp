<%@ page language="java" contentType="text/html; charset=UTF-8"
	pageEncoding="UTF-8"%>
<%@ page import="java.util.*"%>
<%@ page import="stock.model.*"%>
<%@ page import="stock.model.dto.*"%>
<%
	String json;
	String data = request.getParameter("id");
	//String type = request.getParameter("type");
	
	StockDAO sDAO = new StockDAO();

	//search2.html에서사용
	if(data==null){
		ArrayList<StockDTO> search = null;
		search = sDAO.getSinfo();
		json = Converter.convertToJson(search);
	} else{
	//result.html에서사용할예정
		StockDTO search = null;
		search = sDAO.getRinfo(data);
		json = Converter.convertToJson(search);
	} 
%>

<%=json %>
