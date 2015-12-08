/*
 * main.cpp
 *
 *  Created on: 9 Nov 2015
 *      Author: coert
 */

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <stddef.h>
#include <cassert>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <vector>

#ifdef _WIN32
#include <Windows.h>
#include <GL/gl.h>
#include <GL/glu.h>
#endif
#ifdef __linux__
#include <GL/freeglut_std.h>
#include <GL/glut.h>
#include <GL/glu.h>
#endif

std::shared_ptr<cv::Mat> matrix;
const static std::string OPENCV_WINDOW = "OpenCV";
const static std::string SCENE_WINDOW = "OpenGL 3D scene";

void update(
		int v)
{
	cv::imshow("OpenCV", *matrix);
	char key = cv::waitKey(10);
	if (key == 27) exit(EXIT_SUCCESS); //we cannot exit the glutMainLoop()

#ifdef __linux__
	glutTimerFunc(v, update, 0);
#endif
}

#ifdef _WIN32
/**
 * Handle all windows keyboard and mouse inputs with WM_ events
 */
LRESULT CALLBACK WndProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	// Rendering and Device Context variables are declared here.
	static HGLRC hRC;
	static HDC hDC;
	LONG lRet = 1;

	switch(message)
	{
		case WM_CREATE: // Window being created
		{
			hDC = GetDC(hwnd); // Get current windows device context

			static PIXELFORMATDESCRIPTOR pfd =
			{
			   sizeof(PIXELFORMATDESCRIPTOR),//size of structure
			   1,//default version
			   PFD_DRAW_TO_WINDOW |//window drawing support
			   PFD_SUPPORT_OPENGL |//opengl support
			   PFD_DOUBLEBUFFER,//double buffering support
			   PFD_TYPE_RGBA,//RGBA color mode
			   32,//32 bit color mode
			   0, 0, 0, 0, 0, 0,//ignore color bits
			   0,//no alpha buffer
			   0,//ignore shift bit
			   0,//no accumulation buffer
			   0, 0, 0, 0,//ignore accumulation bits
			   16,//16 bit z-buffer size
			   0,//no stencil buffer
			   0,//no aux buffer
			   PFD_MAIN_PLANE,//main drawing plane
			   0,//reserved
			   0, 0, 0//layer masks ignored
			};
			int nPixelFormat = ChoosePixelFormat(hDC, &pfd);
			SetPixelFormat(hDC, nPixelFormat, &pfd);

			hRC = wglCreateContext(hDC);
			wglMakeCurrent(hDC, hRC);

			assert(hRC != nullptr);
			assert(hDC != nullptr);
		}
		break;
		case WM_CLOSE:                              // Window is closing
		{
			hDC = GetDC(hwnd);                              // Get current windows device context
			// Deselect rendering context and delete it
			wglMakeCurrent(hDC, NULL);
			wglDeleteContext(hRC);

			// Send quit message to queue
			PostQuitMessage(0);
		}
		break;
		default:
		lRet = long(DefWindowProc(hwnd, message, wParam, lParam));
	}

	return lRet;
}
#endif 

int main(
		int argc, char** argv)
{
#ifdef __linux__
	std::string filename = "~/Pictures/Chrysanthemum.jpg";
#elif defined _WIN32
	std::string filename = "C:\\Users\\Erik\\Pictures\\a.jpg";
#endif
	if (argc > 1) filename = argv[1];
	cv::Mat image = cv::imread(filename);
	assert(!image.empty());
	matrix = std::make_shared<cv::Mat>(image);

	cv::namedWindow(OPENCV_WINDOW, CV_WINDOW_KEEPRATIO);

#ifdef __linux__
	glutInit(&argc, argv);
	glutInitWindowSize(100, 100);
	glutCreateWindow(SCENE_WINDOW.c_str());
	glutTimerFunc(10, update, 0);
	glutMainLoop();
#elif defined _WIN32
	WNDCLASSEX windowClass;//window class
	HWND hwnd;//window handle
	DWORD dwExStyle;//window extended style
	DWORD dwStyle;//window style
	RECT windowRect;

	/*      Screen/display attributes*/
	int width = 100;
	int height = 100;
	int bits = 32;

	windowRect.left =(long)0;               //set left value to 0
	windowRect.right =(long)width;//set right value to requested width
	windowRect.top =(long)0;//set top value to 0
	windowRect.bottom =(long)height;//set bottom value to requested height

	/*      Fill out the window class structure*/
	windowClass.cbSize = sizeof(WNDCLASSEX);
	windowClass.style = CS_HREDRAW | CS_VREDRAW;
	windowClass.lpfnWndProc = WndProc;
	windowClass.cbClsExtra = 0;
	windowClass.cbWndExtra = 0;
	windowClass.hInstance = 0;                //hInstance;
	windowClass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	windowClass.hCursor = LoadCursor(NULL, IDC_ARROW);
	windowClass.hbrBackground = NULL;
	windowClass.lpszMenuName = NULL;
	windowClass.lpszClassName = LPCWSTR(L"Glut");
	windowClass.hIconSm = LoadIcon(NULL, IDI_WINLOGO);

	/*      Register window class*/
	if (!RegisterClassEx(&windowClass))
	{
		return 0;
	}

	dwExStyle = WS_EX_APPWINDOW | WS_EX_WINDOWEDGE;  //window extended style
	dwStyle = WS_OVERLAPPEDWINDOW;//windows style

	AdjustWindowRectEx(&windowRect, dwStyle, FALSE, dwExStyle);

	/*      Class registerd, so now create our window*/
	hwnd = CreateWindowEx(NULL, LPCWSTR(L"Glut"), //class name
		LPCWSTR(L"Test"), //app name
			dwStyle |
			WS_CLIPCHILDREN |
			WS_CLIPSIBLINGS,
			0, 0,//x and y coords
			windowRect.right - windowRect.left,
			windowRect.bottom - windowRect.top,//width, height
			NULL,//handle to parent
			NULL,//handle to menu
			0,//application instance
			NULL);//no xtra params

	/*      Check if window creation failed (hwnd = null ?)*/
	if (!hwnd)
	{
		return 0;
	}

	ShowWindow(hwnd, SW_SHOW);             //display window
	UpdateWindow(hwnd);//update window

	while(true)
	{
		update(0);
	}
#endif

	return EXIT_SUCCESS;
}

