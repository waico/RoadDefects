
import os
import sys

from bottle import route, run, template
from bottle import get, post, request, response
from bottle import static_file
from IPython import embed

from RDD2 import detect

from PIL import Image

import io

def f_proc(img):
	image_np = detect(img)
	new_img = Image.fromarray(image_np)
	b = io.BytesIO()
	new_img.save(b, "PNG")
	b.seek(0)
	return b
	#return img

@route("/result/<filename>")
def h_get_result(filename):
	root = "imgs"
	return static_file(filename, root = root)


@get("/upload")
def h_get_upload():
	f = open("upload.html", "r")
	h = f.read()
	f.close()
	return h


@post("/upload")
def h_post_upload():
	category   = request.forms.get('category')
	upload     = request.files.get('upload')
	name, ext = os.path.splitext(upload.filename)
	print(f"name: {name}\next: {ext}")
	if ext not in ('.png','.jpg','.jpeg'):
		return 'File extension not allowed.'

	#save_path = get_save_path_for_category(category)
	save_path = "imgs"
	print(f"save path: {save_path}")
	#image_bytes = upload.file.read()
	proc_img = f_proc(upload.file)
	proc_image_bytes = proc_img.read()
	response.set_header('Content-type', 'image/' + ext.replace(".", ""))
	return proc_image_bytes
	#embed()
	#upload.save(save_path) # appends upload.filename automatically
	#return 'OK'

@route('/hello/<name>')
def index(name):
	return template('<b>Hello {{name}}</b>!', name=name)

run(host='0.0.0.0', port=8080, debug = True)
