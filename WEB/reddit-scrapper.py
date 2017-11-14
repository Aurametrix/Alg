# -*- coding: utf-8 -*-
    import scrapy
    from bs4 import BeautifulSoup
    from langdetect import detect_langs
    from scrapy.linkextractors import LinkExtractor

    class RedditbotSpider(scrapy.Spider):
        name = 'redditbot'
        allowed_domains = ['reddit.com' , 'www.reddit.com/r/france/']
        start_urls = [
            'https://www.reddit.com/r/france//',
        ]

        #Where to crawl: https://www.reddit.com/r/rance/
        #https://www.reddit.com/r/programmation/
        #https://www.reddit.com/r/paslegorafi/
        #http://www.reddit.com/r/france//
        global utterance
        utterance = 0
        global english_sentence
        english_sentence = 0
        global conversation
        conversation = 0;


        def parse(self, response):
            titles = response.css('.title.may-blank::text').extract()
            comments = response.css('.comments::text').extract()
            links = response.css('li.first a::attr(href)').extract()
            #Give the extracted content row wise
            for item in zip(titles,comments,links):
                #create a dictionary to store the scraped info
                item_splitted = item[1].split();
                number_comment = item_splitted[0];
                scraped_info = {
                'title' : item[0],
                'comments' : number_comment,
                'links' : item[2],
                }
                #yield or give the scraped info to scrapy
                if number_comment != 'comment' and int(number_comment) >= 5:            
                    yield response.follow(item[2], callback=self.thread_parse)

            next_page = response.css('span.next-button a::attr(href)').extract_first()
            print(next_page)
            if next_page is not None:
                yield response.follow(next_page, callback=self.parse)

        def thread_parse(self,response):
            global utterance
            global english_sentence
            global conversation


            threads = response.xpath('//div[@class="commentarea"]/div[@class="sitetable nestedlisting"]/div[contains(@data-subreddit,"rance")]').extract()
            #Gets all the main threads, now each of its children will be a discussion so we can extract all of the username to make a conversation
            file = open('redditcorpus.xml',"a",encoding='utf8')
            #With that all the messages are in order and the authors of the message too
            #message at index x belongs to authors at index x 
            for index, thread in enumerate(threads):
                soup = BeautifulSoup(thread, "html.parser")
                a_names = soup.select("a.author.may-blank")
                a_names = ' '.join(map(str,a_names))

                div_messages = soup.select("div.md")
                div_messages = ' '.join(map(str,div_messages))

                soup = BeautifulSoup(a_names,"html.parser")
                authors = soup.find_all("a")
                authors = [author.text for author in authors]

                #If there is less than 3 people talking its useless to take this data
                if(len(authors) < 3):
                    continue;

                soup = BeautifulSoup(div_messages,"html.parser")
                messages = soup.find_all("div")
                messages = [message.text for message in messages] #unprocessed div block

                eng = 0;
                #Language Detection
                #for i in range(0,len(authors)):
                #   languages = detect_langs(messages[i])
                #    first_language = str(languages[0])
                #    sub_lang = first_language[0:2]
                #    prob = float(first_language[3:7])
                #    if ((sub_lang == 'en')):
                #        english_sentence = english_sentence + 1;
                #        print("NON-FRENCH SENTENCE: " + messages[i])
                #        eng = 1;
                #        break;
                #if(eng == 1):
                #    continue;


                file.write('<s>');
                dict = {};
                uid = 1;
                current_id = -1;
                for ind in range(0,len(authors)):                

                    if(authors[ind] not in dict):
                        dict[authors[ind]] = uid;
                        uid = uid + 1;
                    current_id = dict[authors[ind]]
                    file.write('<utt uid='+str(current_id)+'> '+messages[ind]+'</utt>')  
                    utterance = utterance + 1;

                file.write('</s>\n')
                conversation = conversation + 1;
            file.close()
            print('Number of utterance = %d' %(utterance))
            print('Number of conversation = %d' %(conversation))
